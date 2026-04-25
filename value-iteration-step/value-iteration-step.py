import numpy as np

def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration.

    Parameters:
    values : array-like of shape (S,)
        Current value function V(s)
    transitions : array-like of shape (S, A, S)
        Transition probabilities T(s, a, s')
    rewards : array-like of shape (S, A)
        Immediate rewards R(s, a)
    gamma : float
        Discount factor (0 <= gamma <= 1)

    Returns:
    list of shape (S,)
        Updated value function V'(s)
    """

    V = np.array(values)                
    T = np.array(transitions)        
    R = np.array(rewards)             

    # Compute expected future value:
    # For each (s, a): sum over s' of T(s,a,s') * V(s')
    # Result shape: (S, A)
    expected_future = np.sum(T * V, axis=2)

    # Compute Q-values using Bellman equation:
    # Q(s,a) = R(s,a) + gamma * expected_future
    Q = R + gamma * expected_future

    # Take max over actions for each state
    V_new = np.max(Q, axis=1)

    # Return as Python list (as required)
    return V_new.tolist()