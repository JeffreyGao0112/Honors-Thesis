import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from PeakDetection import find_and_select_peaks


def read_xy_files(folder_path):
    """Reads multiple .dat xy files from a folder and returns the data as a list."""
    data = []
    try:
        for filename in os.listdir(folder_path):
            if filename.endswith(".dat"):  # Or any other relevant extension
                file_path = os.path.join(folder_path, filename)
                try:
                    # Load data, skipping header lines if present
                    xy_data = np.loadtxt(file_path, comments="#") # or comments="%" or however your file denotes comments
                    x_data = xy_data[:, 0]  # First column is x
                    y_data = xy_data[:, 1]  # Second column is y
                    data.append((x_data, y_data))
                except Exception as e:
                    print(f"Error reading file {filename}: {e}")
    except FileNotFoundError:
        print(f"Folder not found: {folder_path}")
    return data

folder_path = "C:/Personal/Honors Thesis/src/Data/"
#folder_path = "C:/Users/jgao0/.vscode/Personal/Honors-Thesis/Data/"  # LAPTOP PATH
xy_data_list = read_xy_files(folder_path)

print(len(xy_data_list))

feature_vectors = []
for data in xy_data_list: 
    feature_vector, _ = find_and_select_peaks(x=data[0], y=data[1])

    print(f"Feature vector shape: {feature_vector.shape}") # Print shape
    print(f"Feature vector: {feature_vector}") # Print the vector itself

    feature_vectors.append(feature_vector)

print(f"Length of feature_vectors before concatenation: {len(feature_vectors)}")
feature_vectors = np.concatenate(feature_vectors, axis=0)
print(f"Shape of feature_vectors after concatenation: {feature_vectors.shape}")


# Data Scaling (Important for DBSCAN)
scaler = StandardScaler()
scaled_feature_vectors = scaler.fit_transform(feature_vectors)

# DBSCAN Clustering
# Tuning eps and min_samples is CRUCIAL. Experiment with these values.
# Given your comment about "very close points," you'll likely need a small eps.

eps_value = 0.5 # Start with a small value and adjust
min_samples_value = 2 # Start with a low value and adjust.

dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
clusters = dbscan.fit_predict(scaled_feature_vectors)

# Visualize the Clusters (2D projection if feature vector is > 2D)
from sklearn.decomposition import PCA

if feature_vectors.shape[1] > 2:  # Project to 2D for visualization if needed
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(scaled_feature_vectors)  # Use scaled data for PCA
else:
    data_2d = scaled_feature_vectors


plt.figure(figsize=(8, 6))

# Color the points by cluster label. -1 means outlier
unique_labels = np.unique(clusters)
colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    # Select data points belonging to the current cluster
    cluster_data = data_2d[clusters == label]
    
    # Plot the cluster data.  Crucially, plot ALL points for each label.
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=color, label=f"Cluster {label}")

plt.title("DBSCAN Clustering of XRPD Features")
plt.xlabel("PC1")  # Update labels if not using PCA
plt.ylabel("PC2")
plt.legend()
plt.show()

# Print cluster assignments (optional)
print("Cluster Assignments:", clusters)

# Analyze Outliers
n_noise = np.sum(clusters == -1)
print(f"Number of outliers: {n_noise}")

# Analyze cluster sizes
for label in unique_labels:
    if label != -1:  # Don't count outliers as a cluster
        cluster_size = np.sum(clusters == label)
        print(f"Size of cluster {label}: {cluster_size}")