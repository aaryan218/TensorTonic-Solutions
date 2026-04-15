import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    n_samples, n_features = X.shape

    # Initialize parameters
    w = np.zeros(n_features)
    b = 0.0 

    for _ in range(steps):
        # Linear combination
        z = np.dot(X, w) + b
        
        # Prediction
        y_pred = _sigmoid(z)  #By directly using the helper function

        # Gradients
        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)

        # Update
        w -= lr * dw
        b -= lr * db

    return w, b

# Idea:
# We predict probabilities using a sigmoid function over a linear model.
# The difference (ŷ - y) tells us how wrong we are.
# This error is propagated back to adjust weights:
#   -> If prediction is too high → decrease weights
#   -> If prediction is too low → increase weights
# Gradient descent updates parameters in the direction that reduces error.