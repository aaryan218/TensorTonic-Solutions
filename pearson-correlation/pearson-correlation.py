import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix without using np.corrcoef.
    """

    # Convert to numpy array
    X = np.asarray(X, dtype=float)

    # Validate input
    if X.ndim != 2 or X.shape[0] < 2:
        return None

    N, D = X.shape

    # Mean center the data
    X_centered = X - np.mean(X, axis=0)

    # Covariance matrix
    cov = (X_centered.T @ X_centered) / (N - 1)

    # Standard deviations
    std = np.sqrt(np.diag(cov))

    # Denominator matrix
    denom = np.outer(std, std)

    # Correlation matrix initialized with NaN
    corr = np.full((D, D), np.nan)

    # Avoid division by zero
    valid = denom != 0
    corr[valid] = cov[valid] / denom[valid]

    # Set diagonal to 1 for non-zero variance features
    for i in range(D):
        if std[i] != 0:
            corr[i, i] = 1.0

    return corr