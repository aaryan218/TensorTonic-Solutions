import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    scores = np.array(scores, dtype=float)

    T = scores.shape[-1]

    # Create upper-triangular mask excluding diagonal
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)

    # Copy input to avoid in-place modification
    masked_scores = scores.copy()

    # Apply mask using broadcasting
    masked_scores[..., mask] = mask_value

    return masked_scores