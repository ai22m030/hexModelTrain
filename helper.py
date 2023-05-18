import configparser
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from hex_engine import HexPosition


class HexModel(nn.Module):
    def __init__(self, board_size=7, num_channels=64):
        super(HexModel, self).__init__()
        self.board_size = board_size
        self.input_size = board_size ** 2

        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(self.input_size, self.input_size),
            nn.LogSoftmax(dim=-1),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(self.input_size, num_channels),
            nn.ReLU(),
            nn.Linear(num_channels, 1),
            nn.Tanh(),
        )

    def forward(self, board):
        x = board.view(-1, 1, self.board_size, self.board_size)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value

    def save(self, file_name):
        torch.save(self.state_dict(), file_name)

    def load(self, file_name):
        self.load_state_dict(torch.load(file_name))


def read_config():
    data = {}
    config = configparser.ConfigParser()
    config.read('config.ini')
    for section in config.sections():
        for key in config[section]:
            if key == 'debug':
                if config[section][key] == 'True':
                    data[key] = True
                else:
                    data[key] = False
            elif key == 'size':
                data[key] = int(config[section][key])
            else:
                data[key] = config[section][key]

    return data


def db_connect(conf, train=True):
    if train:
        uri = f"mongodb+srv://{conf['username']}:{conf['password']}@{conf['train_host']}/?retryWrites=true&w=majority"
    else:
        uri = f"mongodb+srv://{conf['username']}:{conf['password']}@{conf['val_host']}/?retryWrites=true&w=majority"

    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi('1'))

    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print(f"Pinged your deployment.")
    except Exception as e:
        print(e)
        exit(1)

    return client


def select_database(conf, train=True):
    client = db_connect(conf, train)
    db = client['gameplay_db']
    return db['gameplays']


def get_replays(gameplays_db):
    replays = gameplays_db.find({})
    return list(replays)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    e_x *= (x != 0)

    x_sum = e_x.sum()

    if x_sum == 0:
        x_sum = 1.0

    return e_x / x_sum


def prepare_data(replays):
    env = HexPosition()
    boards = {}
    policies = {}
    values = {}

    for replay in replays:
        gameplay = replay['gameplay']
        moves = replay['moves']
        winner = replay['winner']

        for idx, move in enumerate(moves):
            if idx % 2 != 0:
                env.board = gameplay[idx]
                gameplay[idx] = env.recode_black_as_white()
                move = env.recode_coordinates(move)

            hashable_gameplay = pickle.dumps(gameplay[idx])
            if hashable_gameplay not in boards:
                boards[hashable_gameplay] = np.array(gameplay[idx], dtype=np.float32)

            if hashable_gameplay not in policies:
                policy = np.zeros(env.size ** 2)
            else:
                policy = policies[hashable_gameplay]

            if winner == 1 and idx % 2 == 0:
                policy[move[0] * env.size + move[1]] += 1
            elif winner == -1 and idx % 2 != 0:
                policy[move[0] * env.size + move[1]] += 1
            else:
                policy[move[0] * env.size + move[1]] += -1

            policies[hashable_gameplay] = policy
            values[hashable_gameplay] = 1

    for gameplay, policy in policies.items():
        negative_values_count = np.sum(policy < 0)  # Count negative values in policy
        positive_values_count = np.sum(policy > 0)  # Count positive values in policy

        if np.sum(policy) < 0 and negative_values_count > positive_values_count:
            # If the sum of policy values is negative and there are more negative than positive values
            policies[gameplay] = -policies[gameplay]  # Invert the policy probabilities
            values[gameplay] = -1  # Assign a value of -1 to the losing state

    boards = torch.tensor(np.stack(list(boards.values())), dtype=torch.float32)
    policies = torch.tensor(np.stack([softmax(v) for v in policies.values()]), dtype=torch.float32)
    values = torch.tensor(np.stack(list(values.values())).reshape(-1, 1), dtype=torch.float32)

    return boards, policies, values


def train_hex_model(model, train_data, val_data, epochs=100, batch_size=32, learning_rate=1e-4, weight_decay=1e-4,
                    patience=35, lr_reduction_factor=0.5):
    boards_train, policies_train, values_train = train_data
    boards_val, policies_val, values_val = val_data

    # Convert values tensors to float32
    values_train = values_train.clone().detach()
    values_val = values_val.clone().detach()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=lr_reduction_factor, verbose=True)

    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    train_dataset = torch.utils.data.TensorDataset(boards_train, policies_train, values_train)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(boards_val, policies_val, values_val)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    epochs_since_best = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_data_loader:
            board, policy_target, value_target = batch
            optimizer.zero_grad()

            policy_pred, value_pred = model(board)

            policy_loss = policy_loss_fn(policy_pred, policy_target)
            value_loss = value_loss_fn(value_pred, value_target)
            loss = policy_loss + value_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_data_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_batch in val_data_loader:
                val_board, val_policy_target, val_value_target = val_batch
                val_policy_pred, val_value_pred = model(val_board)
                val_policy_loss = policy_loss_fn(val_policy_pred, val_policy_target)
                val_value_loss = value_loss_fn(val_value_pred, val_value_target)
                val_loss += val_policy_loss.item() + val_value_loss.item()

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        val_loss /= len(val_data_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the model with the best validation loss
            model.save('best_model.pt')

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    return model
