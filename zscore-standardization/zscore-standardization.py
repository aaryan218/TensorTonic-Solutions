import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    Standardize X: (X - mean)/std.
    If 2D and axis=0 → per column
    If 2D and axis=1 → per row
    Returns np.ndarray (float)
    """
    X = np.asarray(X, dtype=float)

    # 1D case
    if X.ndim == 1:
        mean = X.mean()
        std = X.std()
        return (X - mean) / (std + eps)

    # 2D case
    elif X.ndim == 2:
        mean = X.mean(axis=axis, keepdims=True)
        std = X.std(axis=axis, keepdims=True)
        return (X - mean) / (std + eps)

    else:
        raise ValueError("Input must be 1D or 2D array")