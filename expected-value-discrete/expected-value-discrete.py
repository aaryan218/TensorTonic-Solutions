import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    
    # Ensure shapes match
    if x.shape != p.shape:
        raise ValueError("x and p must have the same shape")
    
    # Ensure probabilities sum to 1 (within tolerance)
    if not np.isclose(np.sum(p), 1.0, atol=1e-6):
        raise ValueError("Probabilities must sum to 1")
    
    # Compute expected value
    return float(np.sum(x * p))