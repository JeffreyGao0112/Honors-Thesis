import numpy as np
import joblib # For loading the scaler
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import silhouette_score # For evaluating clustering

# 1. Load the encoded data:
encoded_data = np.load("encoded_data.npy")

# 2. Perform DBSCAN clustering:
# Experiment with eps and min_samples. These are VERY data-dependent!
scaler = StandardScaler() # Create a new scaler specifically for the encoded data
encoded_data_scaled = scaler.fit_transform(encoded_data) # Fit and transform

# 2. Perform DBSCAN clustering on the *scaled* data:
for eps in [0.000001, 0.000002, 0.000005, 0.00001]:  # Try smaller eps values first
    for min_samples in [2, 3, 4, 5]:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(encoded_data_scaled)  # Use scaled data!

        n_clusters = len(set(cluster_labels)) - 1
        n_noise = list(cluster_labels).count(-1)

        if n_clusters > 0:
          try:
            silhouette_avg = silhouette_score(encoded_data_scaled, cluster_labels) # Use scaled data!
            print(f"eps={eps}, min_samples={min_samples}, Clusters={n_clusters}, Noise={n_noise}, Silhouette={silhouette_avg}")
          except ValueError:
            print(f"eps={eps}, min_samples={min_samples}, Clusters={n_clusters}, Noise={n_noise}, Silhouette not defined")
        else:
          print(f"eps={eps}, min_samples={min_samples}, No clusters found")

# 3. Analyze and evaluate the results:
n_clusters = len(set(cluster_labels)) - 1  # Subtract 1 for noise (-1 label)
print(f"Number of clusters: {n_clusters}")

# Check for any points not in a cluster
n_noise = list(cluster_labels).count(-1)
print(f"Number of noise points: {n_noise}")

# Calculate silhouette score (if applicable - only for clusters with at least 2 points)
if n_clusters > 1:
    try:
      silhouette_avg = silhouette_score(encoded_data, cluster_labels)
      print(f"Silhouette Score: {silhouette_avg}")
    except ValueError:
      print("Silhouette score not defined when a cluster has only one sample.")



# 4. Use the cluster labels for further analysis or as input to a classifier:
# Now you can use cluster_labels to understand your data or as targets for a classifier.


# 5. (If you want to create new labels from the clusters)
if n_clusters > 0:
  new_labels = np.zeros(len(cluster_labels), dtype=int)
  for i, label in enumerate(cluster_labels):
    if label != -1: # Ignore noise points
      new_labels[i] = label + 1 # Clusters need to start at 1 for some classifiers
else:
  print("No clusters were found. No new labels were created.")

print(new_labels)