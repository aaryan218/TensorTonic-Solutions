import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    try:
        arr = np.array(matrix, dtype=float)

        if arr.ndim != 2:
            return None

        if axis not in (None, 0, 1):
            return None

        if norm_type not in ('l1', 'l2', 'max'):
            return None

        if norm_type == 'l1':
            norms = np.sum(np.abs(arr), axis=axis, keepdims=True)
        elif norm_type == 'l2':
            norms = np.sqrt(np.sum(arr ** 2, axis=axis, keepdims=True))
        else:  # max norm
            norms = np.max(np.abs(arr), axis=axis, keepdims=True)

        # Prevent division by zero
        norms = np.where(norms == 0, 1, norms)

        return arr / norms

    except Exception:
        return None