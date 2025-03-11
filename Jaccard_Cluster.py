import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def filter_data(file_path, lower_bound=4, upper_bound=30):
    """
    Reads a .dat file and returns the filtered data (as a NumPy array)
    with only the rows where lower_bound <= x <= upper_bound.
    """
    try:
        data = np.loadtxt(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    mask = (data[:, 0] >= lower_bound) & (data[:, 0] <= upper_bound)
    filtered_data = data[mask]
    return filtered_data

def process_file(filepath, height_threshold_percentile, prominence):
    """
    Process a .dat file to find peaks and return a vector.
    The vector is defined as:
      - First element: the x-coordinate of the first (left-most) peak.
      - Following elements: the x-distance of each subsequent peak from the first peak.
    Assumes the file has two columns: x and y.
    """
    try:
        data = filter_data(filepath)
        x = data[:, 0]
        y = data[:, 1]
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    height_threshold = np.percentile(y, height_threshold_percentile)
    peaks, _ = find_peaks(y, prominence=prominence, height=height_threshold)
    if peaks.size == 0:
        print(f"No peaks found in {filepath}.")
        return None

    # Get the x-coordinates for the detected peaks and sort them.
    peak_x_coords = np.sort(x[peaks])
    first_peak = peak_x_coords[0]
    differences = peak_x_coords[1:] - first_peak if peak_x_coords.size > 1 else np.array([])
    vector = np.concatenate(([first_peak], differences))
    return vector

def fuzzy_jaccard(v1, v2, tolerance=0.1):
    """
    Compute a fuzzy Jaccard similarity between two vectors (treated as sets)
    where two elements are considered a match if their absolute difference is <= tolerance.
    Returns: matches / (len(v1) + len(v2) - matches)
    """
    list1 = list(v1)
    list2 = list(v2)
    matches = 0
    used = [False] * len(list2)
    for a in list1:
        for i, b in enumerate(list2):
            if not used[i] and abs(a - b) <= tolerance:
                matches += 1
                used[i] = True
                break
    union = len(list1) + len(list2) - matches
    return matches / union if union != 0 else 1.0

def boosted_fuzzy_jaccard(v1, v2, tolerance=0.1, first_thresh=0.45, boost_multiplier=2.0):
    """
    Compute a fuzzy Jaccard similarity between two vectors and then boost the score
    if the first elements are close.
    The base similarity is computed using all elements except the first.
    If the absolute difference between the first elements is within first_thresh,
    the base similarity is multiplied by boost_multiplier (capped at 1.0).
    """
    base_sim = fuzzy_jaccard(v1[1:], v2[1:], tolerance=tolerance)
    if abs(v1[0] - v2[0]) <= first_thresh:
        return min(1.0, base_sim * boost_multiplier)
    else:
        return base_sim

def select_medoid(cluster, vectors, tolerance, first_thresh, boost_multiplier):
    """
    Given a cluster (list of filenames) and the dictionary of vectors, select
    the medoid—the element with the highest total boosted fuzzy Jaccard similarity
    to all other members of the cluster.
    
    Returns:
        medoid_filename (str): The filename corresponding to the medoid.
    """
    best_medoid = None
    best_total_sim = -np.inf
    for candidate in cluster:
        total_sim = 0
        for other in cluster:
            if candidate == other:
                continue
            sim = boosted_fuzzy_jaccard(vectors[candidate], vectors[other],
                                        tolerance=tolerance, first_thresh=first_thresh,
                                        boost_multiplier=boost_multiplier)
            total_sim += sim
        if total_sim > best_total_sim:
            best_total_sim = total_sim
            best_medoid = candidate
    return best_medoid

def cluster_vectors(vectors, sim_threshold=0.5, tolerance=0.1, first_thresh=0.25, boost_multiplier=4):
    """
    Cluster the vectors using the boosted fuzzy Jaccard similarity.
    Each vector is compared against the representative of each cluster,
    which will later be refined to the medoid.
    """
    clusters = []
    for fname, vec in vectors.items():
        if vec is None:
            continue
        assigned = False
        for cluster in clusters:
            # Use the current medoid of the cluster.
            rep = vectors[select_medoid(cluster, vectors, tolerance, first_thresh, boost_multiplier)]
            sim = boosted_fuzzy_jaccard(rep, vec, tolerance=tolerance,
                                        first_thresh=first_thresh, boost_multiplier=boost_multiplier)
            if sim >= sim_threshold:
                cluster.append(fname)
                assigned = True
                break
        if not assigned:
            clusters.append([fname])
    return clusters

def reassign_outliers(vectors, clusters, tolerance, first_thresh, boost_multiplier, sim_threshold, outlier_threshold):
    """
    For each cluster, compute the medoid and flag any member whose similarity to the medoid
    is below outlier_threshold. Additionally, clusters that have only one or two elements
    are considered outliers entirely. Then, for each outlier, reassign it to the best matching cluster
    based on the similarity to that cluster's medoid. If no cluster meets sim_threshold, form a new cluster.
    """
    outliers = []
    cleaned_clusters = []
    
    # Identify outliers using medoid-based similarity.
    for cluster in clusters:
        if len(cluster) == 0:
            continue
        # If the cluster is too small, treat all its members as outliers.
        if len(cluster) <= 2:
            outliers.extend(cluster)
            continue
        medoid = select_medoid(cluster, vectors, tolerance, first_thresh, boost_multiplier)
        new_cluster = [medoid]  # Keep the medoid.
        for fname in cluster:
            if fname == medoid:
                continue
            sim = boosted_fuzzy_jaccard(vectors[medoid], vectors[fname],
                                        tolerance=tolerance, first_thresh=first_thresh,
                                        boost_multiplier=boost_multiplier)
            if sim < outlier_threshold:
                outliers.append(fname)
            else:
                new_cluster.append(fname)
        cleaned_clusters.append(new_cluster)
    
    # Reassign each outlier.
    for fname in outliers:
        best_sim = -1
        best_cluster = None
        for cluster in cleaned_clusters:
            medoid = vectors[select_medoid(cluster, vectors, tolerance, first_thresh, boost_multiplier)]
            sim = boosted_fuzzy_jaccard(medoid, vectors[fname],
                                        tolerance=tolerance, first_thresh=first_thresh,
                                        boost_multiplier=boost_multiplier)
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster
        if best_sim >= sim_threshold:
            best_cluster.append(fname)
        else:
            cleaned_clusters.append([fname])
    
    return cleaned_clusters


def reassign_outliers_iterative(vectors, clusters, tolerance, first_thresh, boost_multiplier,
                                sim_threshold, initial_outlier_threshold, max_iterations=5, expansion_factor=0.9):
    """
    Iteratively reassign outliers using medoid-based comparisons.
    In each iteration, the outlier threshold is tightened by multiplying with expansion_factor.
    
    Args:
        vectors: Dictionary mapping filename to vector.
        clusters: Initial clusters (list of lists of filenames).
        tolerance, first_thresh, boost_multiplier, sim_threshold: Parameters used in similarity.
        initial_outlier_threshold: Starting threshold for marking outliers.
        max_iterations (int): Maximum number of iterative reassignments.
        expansion_factor (float): Factor by which the outlier threshold is tightened each iteration.
    
    Returns:
        clusters: Refined clusters after iterative outlier reassignment.
    """
    current_threshold = initial_outlier_threshold
    for i in range(max_iterations):
        new_clusters = reassign_outliers(vectors, clusters, tolerance, first_thresh, boost_multiplier,
                                         sim_threshold, current_threshold)
        if new_clusters == clusters:
            print(f"No changes in iteration {i+1}, stopping early.")
            break
        clusters = new_clusters
        print(f"Iteration {i+1}: Outlier threshold tightened to {current_threshold:.3f}")
        current_threshold *= expansion_factor
    return clusters

def merge_similar_clusters(clusters, vectors, tolerance, first_thresh, boost_multiplier, merge_threshold):
    """
    Merge clusters whose medoids are very similar.
    For each pair of clusters, compute the medoid of each and then the boosted fuzzy Jaccard similarity
    between these medoids. If the similarity exceeds merge_threshold, merge the clusters.
    
    This process is repeated iteratively until no more merges occur.
    
    Args:
        clusters: List of clusters (each cluster is a list of filenames).
        vectors: Dictionary mapping filename to vector.
        tolerance, first_thresh, boost_multiplier: Parameters for boosted_fuzzy_jaccard.
        merge_threshold: If the similarity between two cluster medoids is >= merge_threshold, merge them.
    
    Returns:
        merged_clusters: Refined list of clusters after merging.
    """
    merged = True
    while merged:
        merged = False
        new_clusters = []
        skip_indices = set()
        n = len(clusters)
        for i in range(n):
            if i in skip_indices:
                continue
            cluster_i = clusters[i].copy()
            medoid_i = vectors[select_medoid(cluster_i, vectors, tolerance, first_thresh, boost_multiplier)]
            for j in range(i+1, n):
                if j in skip_indices:
                    continue
                cluster_j = clusters[j]
                medoid_j = vectors[select_medoid(cluster_j, vectors, tolerance, first_thresh, boost_multiplier)]
                sim = boosted_fuzzy_jaccard(medoid_i, medoid_j, tolerance=tolerance, 
                                            first_thresh=first_thresh, boost_multiplier=boost_multiplier)
                if sim >= merge_threshold:
                    # Merge cluster_j into cluster_i.
                    cluster_i.extend(cluster_j)
                    skip_indices.add(j)
                    merged = True
            new_clusters.append(cluster_i)
        clusters = new_clusters
    return clusters

def plot_raw_data(fname, data_directory, output_dir, height_threshold_percentile, prominence):
    """
    Plot the raw x,y data from the file and overlay the peaks detected.
    """
    file_path = os.path.join(data_directory, fname)
    try:
        data = np.loadtxt(file_path)
        x = data[:, 0]
        y = data[:, 1]
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    plt.figure()
    plt.plot(x, y, linestyle='-')
    height_threshold = np.percentile(y, height_threshold_percentile)
    peaks, _ = find_peaks(y, prominence=prominence, height=height_threshold)
    plt.plot(x[peaks], y[peaks], 'ro', markersize=5, label="Peaks")
    
    plt.title(f"Raw Data for {fname}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    
    base = os.path.splitext(fname)[0]
    save_path = os.path.join(output_dir, f"{base}.png")
    plt.savefig(save_path)
    plt.close()

def main():
    #######################################################
    sim_threshold = 0.1        # Overall similarity threshold for clustering. Lower = more loose
    tolerance = 0.08
    height_threshold_percentile = 50
    prominence = 120
    boost_multiplier = 10
    first_threshold = 0.25       # How close the first peak needs to be to get the boost.
    outlier_threshold = 0.06    # Starting outlier threshold.
    max_iterations = 3          # Maximum number of iterative reassignments.
    expansion_factor = 1.25          # Expand outlier threshold by a factor each iteration.
    merge_threshold = 0.9       # Similarity threshold for merging clusters. High = tighter
    #######################################################
    
    # Specify the directory containing your .dat files.
    data_directory = "C:/Personal/Honors Thesis/src/Data3/"  # <-- Change this to your actual path
    data_directory = os.path.abspath(data_directory)
    
    # Create the Clusters folder one directory up from the data folder.
    parent_dir = os.path.abspath(os.path.join(data_directory, os.pardir))
    output_base_dir = os.path.join(parent_dir, "Clusters")
    
    # Overwrite the Clusters folder if it already exists.
    if os.path.exists(output_base_dir):
        shutil.rmtree(output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process files to compute vectors.
    vectors = {}
    for filename in os.listdir(data_directory):
        if filename.endswith(".dat"):
            filepath = os.path.join(data_directory, filename)
            vec = process_file(filepath, height_threshold_percentile, prominence)
            vectors[filename] = vec

    # Initial clustering.
    clusters = cluster_vectors(vectors, sim_threshold=sim_threshold, tolerance=tolerance,
                               first_thresh=first_threshold, boost_multiplier=boost_multiplier)
    print(f"Formed {len(clusters)} clusters initially.")

    # Iteratively reassign outliers with progressively tighter thresholds.
    clusters = reassign_outliers_iterative(vectors, clusters, tolerance, first_threshold,
                                           boost_multiplier, sim_threshold, outlier_threshold,
                                           max_iterations=max_iterations, expansion_factor=expansion_factor)
    print(f"Clusters refined to {len(clusters)} clusters after iterative outlier reassignment.")

    # Merge very similar clusters.
    clusters = merge_similar_clusters(clusters, vectors, tolerance, first_threshold, boost_multiplier, merge_threshold)
    print(f"Clusters merged to {len(clusters)} clusters after merging similar clusters.")

    # For each cluster, create a folder and plot the raw data (with peaks) into that folder.
    for i, cluster in enumerate(clusters):
        cluster_dir = os.path.join(output_base_dir, f"cluster_{i+1}")
        os.makedirs(cluster_dir, exist_ok=True)
        for fname in cluster:
            plot_raw_data(fname, data_directory, cluster_dir, height_threshold_percentile, prominence)
            print(f"Plotted raw data for {fname} in {cluster_dir}")

if __name__ == "__main__":
    main()
