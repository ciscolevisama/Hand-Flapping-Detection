import numpy as np

def sliding_window(features, window_size=12, step=1):
    X = []
    for start in range(0, len(features) - window_size + 1, step):
        X.append(features[start:start+window_size])
    return np.array(X)
