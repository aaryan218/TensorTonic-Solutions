# Matrix Transpose

This function manually transposes a matrix by swapping its rows and columns.

NumPy already provides built-in methods for this:

```python
A.T
# or
np.transpose(A)
```

### How this code works

1. Converts the input into a NumPy array.
2. Gets the number of rows and columns.
3. Creates an empty matrix of shape `(columns, rows)`.
4. Uses nested loops to copy each element:
   ```python
   result[j][i] = A[i][j]
   ```
5. Returns the transposed matrix.

### Example

```python
A = [[1, 2, 3],
     [4, 5, 6]]

print(matrix_transpose(A))
```

Output:

```python
[[1 4]
 [2 5]
 [3 6]]
```

> Note: This implementation is for learning purposes. In real applications, use `A.T` or `np.transpose(A)` for better readability and performance.