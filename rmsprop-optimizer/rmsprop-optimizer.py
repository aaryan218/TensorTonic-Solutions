import numpy as np

def rmsprop_step(w, g, s, lr, beta=0.9, eps=1e-8):
    """
    RMSProp Single Update Step

    Approach:
    ----------
    RMSProp is an adaptive learning rate optimizer. Instead of applying the same
    learning rate to all parameters, it scales updates individually based on the
    recent history of gradients.

    Step 1: Maintain a running average of squared gradients
        - We compute an exponential moving average:
              s_t = beta * s_(t-1) + (1 - beta) * (g_t)^2
        - This avoids storing full gradient history (memory efficient).
        - Large gradients over time → larger s → smaller updates.

    Step 2: Scale the gradient using the accumulator
        - Parameters are updated as:
              w_t = w_(t-1) - (lr / (sqrt(s_t) + eps)) * g_t
        - sqrt(s_t) acts like normalization.
        - eps prevents division by zero.

    Key Intuition:
        - If a parameter has consistently large gradients → reduce its step size.
        - If gradients are small → allow larger updates.
        - This stabilizes training and speeds up convergence.

    Why vectorized:
        - All operations are element-wise (NumPy broadcasting).
        - No loops → efficient for high-dimensional data (up to 1e5 parameters).

    Edge Cases:
        - If gradient is zero → no parameter update.
        - Accumulator still decays via beta.

    Parameters:
        w    : current weights (numpy array)
        g    : current gradients (same shape as w)
        s    : running squared gradient accumulator (same shape)
        lr   : learning rate (η)
        beta : decay factor (β), controls memory of past gradients
        eps  : small constant for numerical stability

    Returns:
        nw : updated weights
        ns : updated accumulator
    """
    w = np.array(w, dtype=float)
    g = np.array(g, dtype=float)
    s = np.array(s, dtype=float)

    ns = beta * s + (1 - beta) * (g ** 2)

    nw = w - (lr / (np.sqrt(ns) + eps)) * g

    return nw, ns