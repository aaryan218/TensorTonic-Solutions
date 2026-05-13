import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    x = np.array(x , dtype = float)
    y = np.array(y , dtype = float)
    if x.shape != y.shape:
        raise ValueError()
    diff = np.abs(x - y)
    a = np.sum(diff)
    return a