import numpy as np

def minmax_scale(x, axis=0, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0 (default), scale per column.
    Return np.ndarray (float).
    """
    x = np.array(x, dtype=float)
    mn = x.min(axis=axis, keepdims=True)
    mx = x.max(axis=axis, keepdims=True)
    return ((x - mn) / (mx - mn + eps)).tolist()
    