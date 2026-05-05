def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    if len(y_true) != len(y_pred):
        raise ValueError( )
    
    n = len(y_true)
    if n == 0:
        return 0.0
    
    correct = 0
    for t, p in zip(y_true, y_pred):
        if t == p:
            correct += 1
    
    return correct / n