import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply a 4x4 homogeneous transformation matrix to 3D point(s).

    Parameters:
    T : array-like of shape (4,4)
        Homogeneous transformation matrix [R | t]
    points : array-like of shape (3,) or (N,3)
        Single 3D point or batch of points

    Returns:
    Transformed point(s) with same shape as input:
        (3,) for single point or (N,3) for batch
    """

    # Convert inputs to NumPy arrays (handles lists safely)
    T = np.asarray(T)
    points = np.asarray(points)

    # Extract rotation matrix (top-left 3x3)
    R = T[:3, :3]

    # Extract translation vector (top-right 3x1)
    t = T[:3, 3]

    # Case 1: Single point (shape: (3,))
    if points.ndim == 1:
        # Apply transformation: p' = R*p + t
        return R @ points + t

    # Case 2: Batch of points (shape: (N,3))
    else:
        # Multiply all points with R^T (vectorized)
        # Then add translation (broadcasted automatically)
        return points @ R.T + t