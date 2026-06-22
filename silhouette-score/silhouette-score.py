import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)

    n = len(X)
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        raise ValueError("At least 2 clusters are required.")

    # Pairwise Euclidean distance matrix
    diff = X[:, None, :] - X[None, :, :]
    D = np.sqrt(np.sum(diff**2, axis=2))

    # a(i): mean distance to points in the same cluster
    a = np.zeros(n)

    for c in unique_labels:
        mask = labels == c
        m = mask.sum()

        if m == 1:
            a[mask] = 0.0
        else:
            a[mask] = D[np.ix_(mask, mask)].sum(axis=1) / (m - 1)

    # b(i): minimum mean distance to any other cluster
    b = np.full(n, np.inf)

    for c in unique_labels:
        mask_c = labels == c

        for other in unique_labels:
            if other == c:
                continue

            mask_o = labels == other
            mean_dist = D[np.ix_(mask_c, mask_o)].mean(axis=1)
            b[mask_c] = np.minimum(b[mask_c], mean_dist)

    # Silhouette coefficient for each sample
    s = (b - a) / np.maximum(a, b)

    return float(np.mean(s))