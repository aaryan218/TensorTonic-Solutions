import numpy as np

def softmax(x):
    """
    Compute the Softmax activation function.

    Parameters
    ----------
    x : np.ndarray
        Input NumPy array.
        Can be:
        - 1D array -> returns softmax probabilities for the vector
        - 2D array -> computes softmax row-wise

    Returns
    -------
    np.ndarray
        Array of same shape as input containing probabilities.
        Values will sum to 1:
        - Entire vector for 1D input
        - Each row for 2D input
    """

    # Convert input to NumPy array with float type
    # Using float64 improves numerical precision
    x = np.array(x, dtype=np.float64)

    # ============================================================
    # CASE 1: Input is a 1D vector
    # Example:
    # x = [1, 2, 3]
    # ============================================================
    if x.ndim == 1:

        # Numerical stability trick:
        # Subtract the maximum value from all elements.
        #
        # Why?
        # exp(1000) is extremely large and may overflow.
        # Subtracting max keeps values small while preserving
        # the final probability distribution.
        #
        # Example:
        # [1, 2, 3] -> [-2, -1, 0]
        x_shifted = x - np.max(x)

        # Compute exponentials of shifted values
        exp_x = np.exp(x_shifted)

        # Compute sum of exponentials
        exp_sum = np.sum(exp_x)

        # Normalize:
        # Divide each exponential by total sum
        probabilities = exp_x / exp_sum

        return probabilities

    # ============================================================
    # CASE 2: Input is a 2D matrix
    # Example:
    # [[1, 2, 3],
    #  [0, 0, 0]]
    #
    # Softmax is applied ROW-WISE
    # ============================================================
    elif x.ndim == 2:

        # Find maximum value in each row
        #
        # axis=1       -> operate across columns
        # keepdims=True -> keeps shape compatible for broadcasting
        #
        # Example:
        # [[1,2,3],
        #  [0,0,0]]
        #
        # max per row:
        # [[3],
        #  [0]]
        row_max = np.max(x, axis=1, keepdims=True)

        # Subtract row-wise maximum from each row
        x_shifted = x - row_max

        # Compute exponentials
        exp_x = np.exp(x_shifted)

        # Sum exponentials row-wise
        #
        # keepdims=True ensures division broadcasts correctly
        exp_sum = np.sum(exp_x, axis=1, keepdims=True)

        # Normalize each row
        probabilities = exp_x / exp_sum

        return probabilities

    # ============================================================
    # CASE 3: Unsupported dimensions
    # ============================================================
    else:
        raise ValueError()