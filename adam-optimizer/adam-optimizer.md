# Adam Optimizer — NumPy Implementation

This project contains a simple implementation of **one update step of the Adam optimizer** using NumPy.

Adam (*Adaptive Moment Estimation*) is a popular optimization algorithm used in machine learning and deep learning. It combines the benefits of momentum and adaptive learning rates, making training faster and more stable.

## How It Works

Adam keeps track of two moving averages:

* **First moment (`m`)** → the average of past gradients (momentum)
* **Second moment (`v`)** → the average of squared gradients (variance)

For each training step:

1. Update the first moment estimate.
2. Update the second moment estimate.
3. Apply bias correction to both estimates.
4. Update the parameters using the corrected values.

Because `m` and `v` start at zero, they are initially biased toward zero. Adam corrects this bias using the timestep `t`, which improves performance during the early stages of training.

## Function

```python
adam_step(param, grad, m, v, t,
          lr=1e-3,
          beta1=0.9,
          beta2=0.999,
          eps=1e-8)
```

### Parameters

| Parameter | Description                                         |
| --------- | --------------------------------------------------- |
| `param`   | Current parameter values                            |
| `grad`    | Gradient of the loss with respect to the parameters |
| `m`       | First moment estimate                               |
| `v`       | Second moment estimate                              |
| `t`       | Current timestep (starting from 1)                  |
| `lr`      | Learning rate                                       |
| `beta1`   | Momentum smoothing factor                           |
| `beta2`   | Variance smoothing factor                           |
| `eps`     | Small value to prevent division by zero             |

### Returns

```python
(param_new, m_new, v_new)
```

* `param_new` → updated parameters
* `m_new` → updated first moment
* `v_new` → updated second moment

## Example

```python
import numpy as np

param = np.array([1.0, 2.0])
grad = np.array([0.1, -0.2])

m = np.zeros_like(param)
v = np.zeros_like(param)

param, m, v = adam_step(
    param,
    grad,
    m,
    v,
    t=1
)

print(param)
```

## Why Bias Correction Matters

At the beginning of optimization:

```python
m = 0
v = 0
```

This causes the moving averages to underestimate the true gradient statistics.

Adam fixes this by computing:

```python
m_hat = m_new / (1 - beta1**t)
v_hat = v_new / (1 - beta2**t)
```

Without this correction, updates would be smaller than intended during the first few iterations.

## Default Values

The standard Adam hyperparameters are:

```python
lr = 0.001
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
```

These defaults work well for many machine learning tasks and are widely used in popular frameworks such as TensorFlow and PyTorch.

## Notes

* Accepts scalars, Python lists, and NumPy arrays.
* Automatically converts inputs to NumPy arrays.
* Suitable for educational purposes and understanding how Adam works internally.
* Implements a single optimization step, making it easy to integrate into custom training loops.
