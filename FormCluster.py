import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PeakDetection import find_and_select_peaks, bucketize
from Encoder import create_autoencoder
import shutil

def read_xy_files(folder_path):
    """Reads multiple .dat xy files from a folder and returns the data as a list."""
    data = []
    try:
        for filename in os.listdir(folder_path):
            if filename.endswith(".dat"):
                file_path = os.path.join(folder_path, filename)
                try:
                    xy_data = np.loadtxt(file_path, comments="#")
                    x_data = xy_data[:, 0]
                    y_data = xy_data[:, 1]
                    data.append((x_data, y_data, filename))  # Store filename too
                except Exception as e:
                    print(f"Error reading file {filename}: {e}")
    except FileNotFoundError:
        print(f"Folder not found: {folder_path}")
    return data

def plot_and_save_xy(x_data, y_data, filename, save_folder):
    """Plots x,y data from .dat files and saves the plots as images."""
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    plt.figure()
    plt.plot(x_data, y_data)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Plot of {filename}")
    save_path = os.path.join(save_folder, f"{os.path.splitext(filename)[0]}.png")
    plt.savefig(save_path)
    plt.close()

def cluster_and_plot(data_folder, plot_folder, cluster_folder):
    """Clusters data, plots, and moves plots to cluster folders."""
    if os.path.exists(cluster_folder):
        for item in os.listdir(cluster_folder):
            item_path = os.path.join(cluster_folder, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
        print(f"Removed existing cluster folders from: {cluster_folder}")
    
    xy_data_list = read_xy_files(data_folder)

    feature_vectors = []
    filenames = []
    for x_data, y_data, filename in xy_data_list:
        feature_vector = bucketize(x_data, y_data)
        feature_vectors.append(feature_vector)
        filenames.append(filename)

    #MANUAL PADDING
    max_len = max(len(vector) for vector in feature_vectors)  # Find max length
    padded_feature_vectors = np.zeros((len(feature_vectors), max_len), dtype=np.float32)

    for i, vector in enumerate(feature_vectors):
        padded_feature_vectors[i, :len(vector)] = vector

    # Remove rows with only NaN values
    mask = ~np.all(np.isnan(padded_feature_vectors), axis=1)
    padded_feature_vectors = padded_feature_vectors[mask]
    filenames = [filenames[i] for i, m in enumerate(mask) if m]

    
    # Autoencoder
    input_dim = padded_feature_vectors.shape[1]
    autoencoder, encoder_model = create_autoencoder(input_dim, 1)

    # Train the autoencoder (handle NaNs) - Corrected mask usage
    row_valid_mask = ~np.all(np.isnan(padded_feature_vectors), axis=1) # 1D mask
    autoencoder.fit(padded_feature_vectors[row_valid_mask], padded_feature_vectors[row_valid_mask], epochs=50, batch_size=32, verbose=0)

    # Get encoded features
    encoded_features = encoder_model.predict(padded_feature_vectors, verbose=0)
    
    eps_value = 0.02
    min_samples_value = 2

    dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
    clusters = dbscan.fit_predict(encoded_features)

    unique_labels = np.unique(clusters)

    # Create cluster folders
    for label in unique_labels:
        if label == -1:
            cluster_path = os.path.join(cluster_folder, "outliers")  # Outlier folder
        else:
            cluster_path = os.path.join(cluster_folder, f"cluster_{label}")
        
        if not os.path.exists(cluster_path):
            os.makedirs(cluster_path)

    # Plot and move files
    for i, (x_data, y_data, filename) in enumerate(xy_data_list):
        plot_and_save_xy(x_data, y_data, filename, plot_folder)
        plot_filename = f"{os.path.splitext(filename)[0]}.png"
        plot_path = os.path.join(plot_folder, plot_filename)

        cluster_label = clusters[i]
        if cluster_label == -1:
            cluster_path = os.path.join(cluster_folder, "outliers", plot_filename)  # Outlier folder
        else:
            cluster_path = os.path.join(cluster_folder, f"cluster_{cluster_label}", plot_filename)
        shutil.move(plot_path, cluster_path)

    # Print cluster stats
    n_noise = np.sum(clusters == -1)
    print(f"Number of outliers: {n_noise}")

    for label in unique_labels:
        if label != -1:
            cluster_size = np.sum(clusters == label)
            print(f"Size of cluster {label}: {cluster_size}")
    """
    #Visualization 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # Black for outliers
        class_member_mask = (clusters == k)
        xy = encoded_features[class_member_mask]
        ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], c=[col], marker='o', edgecolor='k', s=36)

    ax.set_title("Encoded Features (Encoding Dim = 3) with DBSCAN Clusters")
    ax.set_xlabel("Encoded Feature 1")
    ax.set_ylabel("Encoded Feature 2")
    ax.set_zlabel("Encoded Feature 3")
    plt.show()
    """

if __name__ == "__main__":

    data_folder = "C:/Personal/Honors Thesis/src/Data/"
    plot_folder = "C:/Personal/Honors Thesis/src/Plots/"
    cluster_folder = "C:/Personal/Honors Thesis/src/Clusters/"

    cluster_and_plot(data_folder, plot_folder, cluster_folder)