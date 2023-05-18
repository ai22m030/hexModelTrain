from helper import prepare_data, select_database, read_config, get_replays, train_hex_model, HexModel

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    conf = read_config()

    model = HexModel(conf['size'])
    mcts_model = train_hex_model(model, prepare_data(get_replays(select_database(conf))),
                                 prepare_data(get_replays(select_database(conf, False))))

    mcts_model.save("last_model.pt")
