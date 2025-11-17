import numpy as np


def initialize_weights_normal(mean, std, shape):
    return np.random.normal(mean, std, shape)

def relu(z):
    return np.maximum(0, z)

def forward_pass(X, params): 
    # calculating z
    zh = X @ params['wh'] + params['bh']
    #calculating a
    ah = relu(zh)

    return 0