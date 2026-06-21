import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Ensure numpy arrays (handles scalars, lists, arrays)
    param = np.asarray(param, dtype=float)
    grad = np.asarray(grad, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)

    # Step 1: First moment
    m_new = beta1 * m + (1 - beta1) * grad

    # Step 2: Second moment
    v_new = beta2 * v + (1 - beta2) * (grad ** 2)

    # Step 3: Bias correction
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)

    # Step 4: Parameter update
    param_new = param - lr * (m_hat / (np.sqrt(v_hat) + eps))

    return param_new, m_new, v_new


'''
At early steps, m and v are biased toward zero because they start at 0.

So we correct them using timestep t:

Divide m by (1 - beta1^t)
Divide v by (1 - beta2^t)

This step is what makes Adam work properly in early iterations.


** beta1 for momentum smoothing
** beta2 for variance smoothing
'''