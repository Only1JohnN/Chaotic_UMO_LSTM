# System parameters
UMO_PARAMS = {
    'a': 0.5,
    'b': 0,
    'c': 0.08,
    'd': 0.01,
    'initial_state': [1, 0, 0, 0],
    't_span': [0, 1000],
    'num_points': 10000
}

# LSTM parameters
LSTM_CONFIG = {
    'window_size': 50,
    'train_test_split': 0.8,
    'epochs': 100,
    'batch_size': 32,
    'lstm_units': [64, 32]
}