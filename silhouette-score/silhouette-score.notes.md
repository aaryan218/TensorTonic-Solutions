# Compute Silhouette Score
## Intuition

The Silhouette Score measures how well data points are clustered by comparing:

- **Cohesion**: How close a point is to other points in its own cluster.
- **Separation**: How far a point is from points in the nearest neighboring cluster.

For each point:

- **a(i)** = Average distance to all other points in the same cluster.
- **b(i)** = Minimum average distance to points in any other cluster.

The silhouette value is:

[
s(i)=\frac{b(i)-a(i)}{\max(a(i), b(i))}
]

The final Silhouette Score is the average of all individual silhouette values.
## Interpretation

- **+1** → Point is well matched to its own cluster and far from others.
- **0** → Point lies near the boundary between clusters.
- **-1** → Point may be assigned to the wrong cluster.
## Algorithm

1. Compute pairwise Euclidean distances between all points.
2. For each point, calculate its intra-cluster distance **a(i)**.
3. Compute the average distance from the point to every other cluster.
4. Select the smallest of these averages as **b(i)**.
5. Calculate the silhouette value using the formula above.
6. Return the mean silhouette value across all points.
## Complexity

- Time Complexity: **O(n²)**
- Space Complexity: **O(n²)**
## Example

Input:

```text
X = [[0,0],[0,1],[1,0],[5,5],[5,6],[6,5]]
labels = [0,0,0,1,1,1]

```

Output:

```text
≈ 0.84

```

A score close to **1** indicates that the clusters are compact and well separated.