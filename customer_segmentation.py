
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

np.random.seed(42)
X, _ = make_blobs(
    n_samples=200,
    centers=5,
    cluster_std=0.8,
    n_features=2,
    random_state=42
)

df = pd.DataFrame(X, columns=['Annual_Income', 'Spending_Score'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)


inertias = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, 'bo-', linewidth=2)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.savefig('elbow_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# k = 5
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)


sil_score = silhouette_score(X_scaled, df['Cluster'])
print(f"Silhouette Score for k={optimal_k}: {sil_score:.4f}")


centers_scaled = kmeans.cluster_centers_
centers_original = scaler.inverse_transform(centers_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df, x='Annual_Income', y='Spending_Score',
    hue='Cluster', palette='viridis', s=80, alpha=0.7
)
plt.scatter(
    centers_original[:, 0], centers_original[:, 1],
    c='red', marker='X', s=200, label='Centroids'
)
plt.title(f'Customer Segments (k={optimal_k}) – Silhouette = {sil_score:.3f}')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.savefig('cluster_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nCluster sizes:")
print(df['Cluster'].value_counts().sort_index())

print("\nCluster centroids (original scale):")
centroid_df = pd.DataFrame(
    centers_original,
    columns=['Annual_Income', 'Spending_Score']
)
print(centroid_df)