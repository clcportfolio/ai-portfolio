# Clustering Crash Course

Prepared for the **BNSF Railway Data Scientist I/II** assessment.

## Notebooks

| Notebook | Topics |
|---|---|
| `01_clustering_fundamentals.ipynb` | K-Means, Elbow Method, Silhouette Score, Hierarchical Clustering |
| `02_clustering_advanced.ipynb` | DBSCAN, cluster profiling, feature engineering, evaluation deep-dive |

## Key Concepts to Know Cold

### Clustering Algorithms

| Algorithm | Type | Key Params | Pros | Cons |
|---|---|---|---|---|
| K-Means | Centroid-based | `k` (# clusters) | Fast, scalable, interpretable | Must choose k; assumes spherical clusters |
| Hierarchical | Linkage-based | `linkage`, `distance` | No need to pre-specify k; dendrogram | Slow on large data O(n²) |
| DBSCAN | Density-based | `eps`, `min_samples` | Finds arbitrary shapes; handles noise/outliers | Sensitive to params; struggles with varying density |
| GMM | Probabilistic | `n_components` | Soft assignments; handles elliptical clusters | Assumes Gaussian; slower |

### Choosing K (K-Means)
1. **Elbow Method**: plot inertia vs k — look for the "elbow" where adding more clusters gives diminishing returns
2. **Silhouette Score**: ranges from -1 to +1; higher is better; measures how similar a point is to its own cluster vs neighbors
3. **Gap Statistic**: compares inertia to expected under random distribution
4. **Domain knowledge**: often the most practical guide

### Evaluation Metrics
| Metric | Range | Notes |
|---|---|---|
| Inertia (WCSS) | 0 to ∞ | Lower is better; always decreases with more clusters |
| Silhouette Score | -1 to 1 | Higher is better; 0.5+ is reasonable |
| Davies-Bouldin Index | 0 to ∞ | Lower is better; measures cluster overlap |
| Calinski-Harabasz | 0 to ∞ | Higher is better; ratio of between/within cluster variance |

> **Note**: These metrics only work when ground truth labels are unknown. If you have labels, use ARI, NMI, or homogeneity.

### DBSCAN Key Concepts
- **Core point**: has ≥ `min_samples` neighbors within `eps` radius
- **Border point**: within eps of a core point but doesn't meet min_samples itself
- **Noise point** (label = -1): not reachable from any core point
- Choosing `eps`: plot k-NN distance graph, look for the "knee"

### Pre-processing is Critical
- **Always scale features** before clustering (KMeans, DBSCAN use distance metrics)
- Use StandardScaler or MinMaxScaler
- Consider PCA for dimensionality reduction / visualization

### Cluster Profiling (After Clustering)
After assigning labels, analyze each cluster's characteristics:
```python
df.groupby('cluster')[features].agg(['mean', 'median', 'std'])
```
This is often what stakeholders care about most.

## Interview Tips for BNSF Assessment
- Always ask: *"What's the business goal?"* before choosing an algorithm
- Justify your choice of k with evidence (elbow + silhouette)
- Discuss what the clusters **mean** — not just the metrics
- Be ready to handle outliers (DBSCAN handles them naturally)
- Transportation context: freight station segmentation, route clustering, maintenance pattern detection, customer behavior grouping
