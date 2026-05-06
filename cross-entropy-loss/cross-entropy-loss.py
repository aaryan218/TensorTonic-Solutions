import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.

    Parameters:
    y_true : array-like of shape (N,)
        True class labels (integers: 0 to C-1)

    y_pred : array-like of shape (N, C)
        Predicted probabilities for each class

    Returns:
    float
        Average cross-entropy loss
    """
    # Convert inputs to NumPy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    N = y_true.shape[0]

    # Get predicted probabilities for correct classes
    probs = y_pred[np.arange(N), y_true]

    # Prevent log(0)
    probs = np.clip(probs, 1e-15, 1.0)

    # Compute loss
    loss = -np.log(probs)

    return np.mean(loss)
