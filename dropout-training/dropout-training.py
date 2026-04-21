import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.array(x)
    if rng is None:
        rand = np.random.random(x.shape)
    else:
        rand = rng.random(x.shape)
    mask = (rand < (1-p)) / (1-p)
    output = x * mask
    return output, mask