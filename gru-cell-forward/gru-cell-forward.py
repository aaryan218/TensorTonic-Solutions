import numpy as np

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """

    # Unpack parameters
    Wz, Uz, bz = params["Wz"], params["Uz"], params["bz"]
    Wr, Ur, br = params["Wr"], params["Ur"], params["br"]
    Wh, Uh, bh = params["Wh"], params["Uh"], params["bh"]

    # Ensure 2D inputs
    x, x_was_1d = _as2d(x, Wz.shape[0])
    h_prev, h_was_1d = _as2d(h_prev, Uz.shape[0])

    # Update gate
    z = _sigmoid(x @ Wz + h_prev @ Uz + bz)

    # Reset gate
    r = _sigmoid(x @ Wr + h_prev @ Ur + br)

    # Candidate hidden state
    h_tilde = np.tanh(
        x @ Wh + (r * h_prev) @ Uh + bh
    )

    # Final hidden state
    h = (1.0 - z) * h_prev + z * h_tilde

    # Restore original shape if inputs were 1D
    if x_was_1d or h_was_1d:
        return h.reshape(-1)

    return h