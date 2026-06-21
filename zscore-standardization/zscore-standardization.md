# Z-Score Standardization (Feature Scaling)

A simple NumPy implementation of **Z-score standardization**, one of the most common preprocessing techniques in machine learning and statistics.

Standardization transforms data so that it has:

* **Mean = 0**
* **Standard Deviation = 1**

This helps machine learning algorithms train more efficiently and prevents features with large scales from dominating the learning process.

---

## What Is Z-Score Standardization?

Given a value (x), its standardized value is computed as:

[
z = \frac{x - \mu}{\sigma}
]

Where:

* (x) = original value
* (\mu) = mean of the data
* (\sigma) = standard deviation of the data

After transformation:

* Values above the mean become positive
* Values below the mean become negative
* The dataset is centered around zero

---

## Function Signature

```python
zscore_standardize(X, axis=0, eps=1e-12)
```

---

## Parameters

| Parameter | Description                                         |
| --------- | --------------------------------------------------- |
| `X`       | Input array (1D or 2D)                              |
| `axis`    | Direction for computing mean and standard deviation |
| `eps`     | Small value added to avoid division by zero         |

---

## Returns

Returns a NumPy array containing the standardized values.

```python
standardized_X
```

The returned array has the same shape as the input.

---

## Supported Inputs

### 1D Arrays

```python
X = np.array([10, 20, 30, 40, 50])
```

The function computes:

* Mean of the entire array
* Standard deviation of the entire array

Then standardizes every value.

---

### 2D Arrays

```python
X = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
```

You can choose whether to standardize:

* Per column (`axis=0`)
* Per row (`axis=1`)

---

## How It Works

### Step 1: Convert Input to NumPy Array

```python
X = np.asarray(X, dtype=float)
```

This ensures:

* Lists work correctly
* Integers are converted to floating-point values
* Mathematical operations behave consistently

---

### Step 2: Compute Mean

For a column:

```python
mean = X.mean(axis=0)
```

Example:

```text
[1, 2]
[3, 4]
[5, 6]
```

Column means:

```text
[3, 4]
```

---

### Step 3: Compute Standard Deviation

```python
std = X.std(axis=0)
```

The standard deviation measures how spread out the values are around the mean.

A larger standard deviation indicates greater variability.

---

### Step 4: Standardize

```python
(X - mean) / std
```

Each value is:

1. Centered by subtracting the mean.
2. Scaled by dividing by the standard deviation.

---

## Example: 1D Input

```python
import numpy as np

X = np.array([10, 20, 30, 40, 50])

Z = zscore_standardize(X)

print(Z)
```

Output:

```text
[-1.414 -0.707 0.000 0.707 1.414]
```

Notice:

* Mean ≈ 0
* Standard deviation ≈ 1

---

## Example: Column-Wise Standardization

```python
X = np.array([
    [1, 10],
    [2, 20],
    [3, 30]
])

Z = zscore_standardize(X, axis=0)
```

Result:

```text
[
 [-1.225 -1.225]
 [ 0.000  0.000]
 [ 1.225  1.225]
]
```

Each column is standardized independently.

---

## Example: Row-Wise Standardization

```python
X = np.array([
    [1, 2, 3],
    [10, 20, 30]
])

Z = zscore_standardize(X, axis=1)
```

Result:

```text
[
 [-1.225  0.000  1.225]
 [-1.225  0.000  1.225]
]
```

Each row is standardized independently.

---

## Why `keepdims=True`?

The implementation uses:

```python
mean = X.mean(axis=axis, keepdims=True)
std = X.std(axis=axis, keepdims=True)
```

This preserves dimensions and enables broadcasting.

Example:

Without `keepdims=True`:

```python
mean.shape
# (3,)
```

With `keepdims=True`:

```python
mean.shape
# (1, 3)
```

This allows NumPy to correctly subtract the mean from every row or column.

---

## Why Add `eps`?

The implementation uses:

```python
(X - mean) / (std + eps)
```

to avoid division by zero.

Consider:

```python
X = [5, 5, 5, 5]
```

Then:

```python
std = 0
```

Without `eps`:

```python
(X - mean) / 0
```

would produce invalid values.

Adding a tiny constant ensures numerical stability.

---

## Error Handling

The function accepts only:

* 1D arrays
* 2D arrays

Inputs with more dimensions raise an error:

```python
ValueError("Input must be 1D or 2D array")
```

Example:

```python
X = np.random.rand(3, 4, 5)

zscore_standardize(X)
```

Output:

```text
ValueError: Input must be 1D or 2D array
```

---

## Time Complexity

Let:

* (n) = number of elements

### Mean Computation

```text
O(n)
```

### Standard Deviation Computation

```text
O(n)
```

### Standardization

```text
O(n)
```

Overall:

```text
O(n)
```

---

## Applications

Z-score standardization is commonly used in:

* Linear Regression
* Logistic Regression
* Support Vector Machines (SVM)
* Neural Networks
* K-Means Clustering
* Principal Component Analysis (PCA)
* Statistical Analysis
* Anomaly Detection

It is especially important for algorithms that rely on distances, gradients, or variance.

---

## Key Takeaways

* Centers data around zero.
* Scales features to unit variance.
* Improves numerical stability during training.
* Prevents large-scale features from dominating smaller ones.
* Supports both 1D and 2D NumPy arrays.
* Includes protection against division-by-zero errors.

This implementation provides a clean and efficient way to perform feature standardization using only NumPy.
