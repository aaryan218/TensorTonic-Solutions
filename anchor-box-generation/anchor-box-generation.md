# Anchor Box Generator

A simple Python implementation for generating **anchor boxes** used in object detection models such as Faster R-CNN, SSD, RetinaNet, and related architectures.

Anchor boxes (also called *priors*) are predefined bounding boxes placed across an image at different sizes and aspect ratios. During training, the model learns how to adjust these anchors to match real objects.

---

## What This Function Does

The `generate_anchors()` function creates anchor boxes for every location in a feature map.

For each grid cell, it:

1. Computes the center position in the original image.
2. Generates anchors at multiple scales.
3. Generates anchors at multiple aspect ratios.
4. Converts each anchor from center coordinates to corner coordinates.
5. Returns all anchors as bounding boxes in `[x1, y1, x2, y2]` format.

---

## Function Signature

```python
generate_anchors(
    feature_size,
    image_size,
    scales,
    aspect_ratios
)
```

---

## Parameters

| Parameter       | Description                                         |
| --------------- | --------------------------------------------------- |
| `feature_size`  | Size of the feature map (e.g., 10 for a 10×10 grid) |
| `image_size`    | Size of the input image                             |
| `scales`        | List of anchor sizes                                |
| `aspect_ratios` | List of width-to-height ratios                      |

---

## Return Value

Returns a list of anchor boxes:

```python
[
    [x1, y1, x2, y2],
    [x1, y1, x2, y2],
    ...
]
```

Each anchor is represented using corner coordinates:

* `x1, y1` → top-left corner
* `x2, y2` → bottom-right corner

---

## Example

```python
anchors = generate_anchors(
    feature_size=4,
    image_size=256,
    scales=[32, 64],
    aspect_ratios=[0.5, 1.0, 2.0]
)

print("Total anchors:", len(anchors))
print("First anchor:", anchors[0])
```

Output:

```text
Total anchors: 96
First anchor: [20.69, 9.37, 43.31, 54.63]
```

---

## How Anchor Placement Works

### Step 1: Compute Stride

The stride determines how feature map locations correspond to positions in the original image.

```python
stride = image_size / feature_size
```

Example:

```python
image_size = 320
feature_size = 10

stride = 32
```

This means neighboring feature cells are 32 pixels apart in image space.

---

### Step 2: Locate Grid Centers

For each grid location:

```python
cx = (j + 0.5) * stride
cy = (i + 0.5) * stride
```

The `+ 0.5` places anchors at the center of each cell rather than at the corner.

For a 4×4 grid:

```text
●────●────●────●
│    │    │    │
●────●────●────●
│    │    │    │
●────●────●────●
│    │    │    │
●────●────●────●
```

Each dot represents an anchor center.

---

### Step 3: Apply Scale and Aspect Ratio

Each center generates multiple anchors.

Given:

```python
scale = s
aspect_ratio = r
```

Width and height are computed as:

```python
w = s * sqrt(r)
h = s / sqrt(r)
```

This preserves the anchor area:

```python
w * h = s²
```

Examples:

| Aspect Ratio | Shape          |
| ------------ | -------------- |
| 0.5          | Tall rectangle |
| 1.0          | Square         |
| 2.0          | Wide rectangle |

---

### Step 4: Convert to Corner Coordinates

The anchor is stored as:

```python
x1 = cx - w / 2
y1 = cy - h / 2

x2 = cx + w / 2
y2 = cy + h / 2
```

Result:

```text
(x1,y1) ┌─────────┐
         │         │
         │    •    │
         │         │
         └─────────┘ (x2,y2)
```

The dot represents the anchor center.

---

## Total Number of Anchors

The total anchor count is:

```python
feature_size²
× number_of_scales
× number_of_aspect_ratios
```

Example:

```python
feature_size = 10
scales = [32, 64, 128]
aspect_ratios = [0.5, 1.0, 2.0]
```

```text
10 × 10 × 3 × 3 = 900 anchors
```

---

## Common Mistakes

### Incorrect Aspect Ratio Formula

Wrong:

```python
w = s * r
h = s / r
```

Correct:

```python
w = s * sqrt(r)
h = s / sqrt(r)
```

Using the wrong formula changes the anchor area and produces distorted boxes.

### Missing the Center Offset

Wrong:

```python
cx = j * stride
cy = i * stride
```

Correct:

```python
cx = (j + 0.5) * stride
cy = (i + 0.5) * stride
```

Without the offset, anchors are shifted toward grid corners.

### Wrong Iteration Order

The implementation uses row-major order:

```python
for i in rows:
    for j in cols:
```

This produces anchors from left-to-right, top-to-bottom and matches most detection pipelines.

---

## Applications

Anchor boxes are commonly used in:

* Faster R-CNN
* SSD (Single Shot Detector)
* RetinaNet
* Region Proposal Networks (RPN)
* Multi-scale object detection systems

They provide predefined object hypotheses that the network refines during training and inference.

---

## License

This implementation is intended for educational purposes and experimentation. Feel free to modify and integrate it into your own object detection projects.
