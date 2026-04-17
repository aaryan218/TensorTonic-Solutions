import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Handle empty input
    if not seqs:
        return np.zeros((0, 0), dtype=int)
    
    # Determine max length
    if max_len is None:
        max_len = max(len(seq) for seq in seqs) if seqs else 0
    
    # Initialize result with pad_value
    result = np.full((len(seqs), max_len), pad_value, dtype=int)
    
    # Fill sequences with truncation if needed
    for i, seq in enumerate(seqs):
        trunc = seq[:max_len]  # truncate if longer
        result[i, :len(trunc)] = trunc  # right padding
    
    return result


'''
Find the target length:
    Either use max_len if given
    Otherwise, take the length of the longest sequence
Create a fixed-size matrix
    Shape = (number of sequences, target length)
    Fill it entirely with pad_value
    Copy each sequence into the matrix
    Put elements from the sequence into the row
# If the sequence is shorter → remaining positions stay as padding
# If the sequence is longer → cut it (truncate) to fit
'''