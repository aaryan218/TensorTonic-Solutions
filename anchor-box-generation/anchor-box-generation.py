import math

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    
    anchors = []
    
    # Step 1: Compute stride (mapping feature grid → image space)
    stride = image_size / feature_size
    
    # Step 2: Iterate over grid cells in row-major order
    for i in range(feature_size):        # rows (y direction)
        for j in range(feature_size):    # cols (x direction)
            
            # Step 3: Compute center of the current grid cell in image coordinates
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride
            
            # Step 4: Generate anchors for each (scale, aspect_ratio) pair
            for s in scales:
                for r in aspect_ratios:
                    
                    # Correct width & height computation
                    w = s * math.sqrt(r)
                    h = s / math.sqrt(r)
                    
                    # Step 5: Convert center format → corner format
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    
                    anchors.append([x1, y1, x2, y2])
    
    return anchors


"""
DETAILED EXPLANATION:

1. STRIDE:
   The stride defines how far apart adjacent feature grid cells are in the original image.
   Example: image_size = 320, feature_size = 10 → stride = 32 pixels.

2. GRID ITERATION (ROW-MAJOR ORDER):
   We iterate first over rows (i), then columns (j). This ensures the output order:
   top-left → top-right → next row → ... (important for consistency in detection pipelines).

3. CENTER COMPUTATION:
   Each grid cell corresponds to a region in the original image.
   The center is offset by 0.5 to place anchors at the middle of each cell:
       cx = (j + 0.5) * stride
       cy = (i + 0.5) * stride

4. SCALE & ASPECT RATIO:
   - 'scale' controls overall size of the anchor.
   - 'aspect_ratio = w/h' controls shape.

   To preserve area while adjusting shape:
       w = s * sqrt(r)
       h = s / sqrt(r)

   This ensures:
       w * h = s^2  (constant area for a given scale)

5. BOX FORMAT:
   Anchors are returned in corner format:
       [x1, y1, x2, y2]
   where:
       (x1, y1) = top-left corner
       (x2, y2) = bottom-right corner

6. TOTAL NUMBER OF ANCHORS:
   feature_size^2 * len(scales) * len(aspect_ratios)

7. COMMON PITFALLS:
   - Using r instead of sqrt(r) → incorrect box shapes
   - Missing +0.5 offset → misaligned anchors
   - Wrong loop order → incorrect output sequence
   - Integer division → precision errors

This function mirrors how anchor priors are generated in detectors like Faster R-CNN and SSD.
"""