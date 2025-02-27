import numpy as np
import os
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
import shutil
from scipy.signal import find_peaks
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

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
                    data.append((x_data, y_data, filename))
                except Exception as e:
                    print(f"Error reading file {filename}: {e}")
    except FileNotFoundError:
        print(f"Folder not found: {folder_path}")
    return data

def calculate_peak_distances(x_data, y_data, prominence, height_percentile=None):
    """Finds peaks and calculates distances between them."""
    peaks, properties = find_peaks(y_data, prominence=prominence, height=0)
    if not peaks.size:
        return []
    if height_percentile is not None:
        peak_heights = properties["peak_heights"]
        height_threshold = np.percentile(peak_heights, height_percentile)
        filtered_peaks = peaks[peak_heights >= height_threshold]
    else:
        filtered_peaks = peaks
    if len(filtered_peaks) < 2:
        return []
    peak_x_values = x_data[filtered_peaks]
    distances = np.diff(peak_x_values).tolist()
    return distances

def create_autoencoder(input_dim, encoding_dim):
    """Creates an autoencoder model."""
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation='relu')(input_layer)
    decoder = Dense(input_dim, activation='sigmoid')(encoder)  # or 'linear'

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    encoder_model = Model(inputs=input_layer, outputs=encoder)  # Separate encoder

    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder_model

def cluster_and_plot_peak_distances_autoencoder(data_folder, plot_folder, cluster_folder, prominence=0.1, height_percentile=None, encoding_dim=10, eps_value=0.1, min_samples_value=3):
    """Clusters data based on peak distances (with autoencoder), plots, and moves plots to cluster folders."""
    # Remove existing cluster folders
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
        distances = calculate_peak_distances(x_data, y_data, prominence, height_percentile)
        feature_vectors.append(distances)
        filenames.append(filename)

    # Pad the feature vectors
    max_len = max(len(vector) for vector in feature_vectors)  # Find max length
    padded_feature_vectors = np.zeros((len(feature_vectors), max_len), dtype=np.float32)

    for i, vector in enumerate(feature_vectors):
        padded_feature_vectors[i, :len(vector)] = vector

    # Remove rows with only zero values (no peaks found)
    mask = ~np.all(padded_feature_vectors == 0, axis=1)
    padded_feature_vectors = padded_feature_vectors[mask]
    filenames = [filenames[i] for i, m in enumerate(mask) if m]

    
    # Autoencoder
    input_dim = padded_feature_vectors.shape[1]
    autoencoder, encoder_model = create_autoencoder(input_dim, encoding_dim)

    # Train the autoencoder (handle NaNs)
    row_valid_mask = ~np.all(np.isnan(padded_feature_vectors), axis=1)
    autoencoder.fit(padded_feature_vectors[row_valid_mask], padded_feature_vectors[row_valid_mask], epochs=50, batch_size=32, verbose=0)

    # Get encoded features
    encoded_features = encoder_model.predict(padded_feature_vectors, verbose=0)
    
    # DBSCAN Clustering
    dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
    clusters = dbscan.fit_predict(encoded_features)

    unique_labels = np.unique(clusters)

    # Create cluster folders (including one for outliers)
    for label in unique_labels:
        if label == -1:
            cluster_path = os.path.join(cluster_folder, "outliers")
        else:
            cluster_path = os.path.join(cluster_folder, f"cluster_{label}")
        if not os.path.exists(cluster_path):
            os.makedirs(cluster_path)

    for i, (x_data, y_data, filename) in enumerate(xy_data_list):
        if mask[i]:
            plot_and_save_xy(x_data, y_data, filename, plot_folder)
            plot_filename = f"{os.path.splitext(filename)[0]}.png"
            plot_path = os.path.join(plot_folder, plot_filename)

            cluster_label = clusters[np.sum(mask[:i])]
            if cluster_label == -1:
                cluster_path = os.path.join(cluster_folder, "outliers", plot_filename)
            else:
                cluster_path = os.path.join(cluster_folder, f"cluster_{cluster_label}", plot_filename)
            shutil.move(plot_path, cluster_path)

    n_noise = np.sum(clusters == -1)
    print(f"Number of outliers: {n_noise}")

    for label in unique_labels:
        if label != -1:
            cluster_size = np.sum(clusters == label)
            print(f"Size of cluster {label}: {cluster_size}")

    #VISUALIZATION
    """
    #3D
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

    #2D
    
    plt.figure()
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # Black for outliers
        class_member_mask = (clusters == k)
        xy = encoded_features[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

    plt.title("Encoded Features (Encoding Dim = 2) with DBSCAN Clusters")
    plt.xlabel("Encoded Feature 1")
    plt.ylabel("Encoded Feature 2")
    plt.show()
    """
    #1D
    plt.figure()
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # Black for outliers
        class_member_mask = (clusters == k)
        xy = encoded_features[class_member_mask]
        plt.plot(xy[:, 0], [0] * len(xy), 'o', markerfacecolor=col, markeredgecolor='k', markersize=6) #plot on x axis, 0 on y axis.

    plt.title("Encoded Features (Encoding Dim = 1) with DBSCAN Clusters")
    plt.xlabel("Encoded Feature 1")
    plt.yticks([]) #remove y ticks, since all points are at 0.
    plt.show()

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

data_folder = "C:/Personal/Honors Thesis/src/Data2/"
plot_folder = "C:/Personal/Honors Thesis/src/Plots/"
cluster_folder = "C:/Personal/Honors Thesis/src/Clusters/"

cluster_and_plot_peak_distances_autoencoder(data_folder, plot_folder, cluster_folder, prominence=80, height_percentile=10, encoding_dim=1, eps_value=0.2, min_samples_value=2)