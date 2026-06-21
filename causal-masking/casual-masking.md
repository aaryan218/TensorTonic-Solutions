# Causal Mask for Self-Attention

A NumPy implementation of a **causal (look-ahead) mask** used in Transformer-based models such as GPT, LLaMA, and other autoregressive language models.

The purpose of a causal mask is to prevent a token from attending to future tokens during self-attention. This ensures that predictions are made using only current and past information.

---

## Why Causal Masking Is Needed

In autoregressive language modeling, the model generates text one token at a time.

When predicting the next token, a position should only have access to:

* Previous tokens
* The current token

It must **not** see future tokens.

For example:

```text
I love deep learning
```

When processing the word **"love"**, the model should not be able to attend to:

```text
deep
learning
```

A causal mask enforces this constraint by replacing attention scores for future positions with a very large negative number.

---

## Function Signature

```python
apply_causal_mask(scores, mask_value=-1e9)
```

---

## Parameters

| Parameter    | Description                                     |
| ------------ | ----------------------------------------------- |
| `scores`     | Attention score matrix with shape `(..., T, T)` |
| `mask_value` | Value used to suppress future positions         |

---

## Returns

```python
masked_scores
```

A NumPy array with the same shape as the input, where future positions have been replaced by `mask_value`.

---

## Input Shape

The function expects:

```python
(..., T, T)
```

Examples:

### Single Attention Matrix

```python
(T, T)
```

Example:

```python
(4, 4)
```

---

### Multi-Head Attention

```python
(num_heads, T, T)
```

Example:

```python
(8, 128, 128)
```

---

### Batched Multi-Head Attention

```python
(batch_size, num_heads, T, T)
```

Example:

```python
(32, 8, 128, 128)
```

The implementation automatically broadcasts the mask across all leading dimensions.

---

## How It Works

### Step 1: Determine Sequence Length

```python
T = scores.shape[-1]
```

For a sequence of length 5:

```text
T = 5
```

---

### Step 2: Create the Causal Mask

```python
mask = np.triu(
    np.ones((T, T), dtype=bool),
    k=1
)
```

For `T = 5`:

```text
[
 [F T T T T]
 [F F T T T]
 [F F F T T]
 [F F F F T]
 [F F F F F]
]
```

Where:

* `True` → future position (mask it)
* `False` → allowed position

---

### Step 3: Copy Scores

```python
masked_scores = scores.copy()
```

This avoids modifying the original tensor.

---

### Step 4: Apply the Mask

```python
masked_scores[..., mask] = mask_value
```

Future positions are replaced by a large negative number.

Example:

Before masking:

```text
[
 [2.1 1.5 0.8 1.2]
 [0.7 2.4 1.1 0.6]
 [1.0 1.3 2.2 0.9]
 [0.5 0.4 1.2 2.7]
]
```

After masking:

```text
[
 [ 2.1 -1e9 -1e9 -1e9]
 [ 0.7  2.4 -1e9 -1e9]
 [ 1.0  1.3  2.2 -1e9]
 [ 0.5  0.4  1.2  2.7]
]
```

---

## Why Use `-1e9`?

After masking, attention scores are passed through Softmax:

```python
softmax(masked_scores)
```

Since:

```text
exp(-1e9) ≈ 0
```

masked positions receive virtually zero attention probability.

Example:

```text
Softmax([2, -1e9, -1e9])
=
[1, 0, 0]
```

This effectively removes future tokens from consideration.

---

## Example Usage

```python
import numpy as np

scores = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

masked = apply_causal_mask(scores)

print(masked)
```

Output:

```text
[
 [ 1.0 -1e9 -1e9]
 [ 4.0  5.0 -1e9]
 [ 7.0  8.0  9.0]
]
```

---

## Visualization

Allowed attention positions:

```text
✓ ✗ ✗ ✗
✓ ✓ ✗ ✗
✓ ✓ ✓ ✗
✓ ✓ ✓ ✓
```

Where:

* ✓ = token can attend
* ✗ = token cannot attend

This creates the familiar lower-triangular attention pattern used in decoder-only Transformers.

---

## Complexity

Let:

```text
T = sequence length
```

### Time Complexity

```text
O(T²)
```

### Space Complexity

```text
O(T²)
```

because a `T × T` mask is created.

---

## Applications

Causal masking is a core component of:

* GPT
* GPT-2
* GPT-3
* GPT-4 style architectures
* LLaMA
* Mistral
* Falcon
* Transformer decoders
* Autoregressive sequence generation

Any model that predicts tokens from left to right typically relies on causal masking.

---

## Key Takeaways

* Prevents information leakage from future tokens.
* Creates a lower-triangular attention pattern.
* Uses large negative values to force future attention probabilities to zero after Softmax.
* Supports batched and multi-head attention tensors through NumPy broadcasting.
* Essential for autoregressive Transformer models.
