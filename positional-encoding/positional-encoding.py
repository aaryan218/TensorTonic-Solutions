import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Positions: (seq_len, 1)
    pos = np.arange(seq_len, dtype=float)[:, np.newaxis]

    # Number of sin/cos frequency pairs (ceil for odd d_model)
    num_freqs = (d_model + 1) // 2

    # Frequency indices: (1, num_freqs)
    i = np.arange(num_freqs, dtype=float)[np.newaxis, :]

    # Compute denominator: base^(2i / d_model)
    div_term = base ** (2 * i / d_model)

    # Angles: (seq_len, num_freqs)
    angles = pos / div_term

    # Initialize output
    pe = np.zeros((seq_len, d_model), dtype=float)

    # Fill even indices with sin
    pe[:, 0::2] = np.sin(angles[:, :pe[:, 0::2].shape[1]])

    # Fill odd indices with cos
    pe[:, 1::2] = np.cos(angles[:, :pe[:, 1::2].shape[1]])

    return pe