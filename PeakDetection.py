import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def find_and_select_peaks(x, y, num_desired_peaks=8, min_prominence_percentile=10):
    """Finds peaks in a signal, adjusting prominence until a minimum number are found.

    Args:
        y: The signal (intensity values).
        x: The corresponding x-values (2-theta).
        num_desired_peaks: The minimum number of peaks to find.
        min_prominence_percentile: The starting percentile for prominence threshold.

    Returns:
        A tuple containing:
            - top_peaks: Indices of the selected peaks.
            - top_peak_prominences: Prominences of the selected peaks.
            - feature_vector: A feature vector of fixed length (2 * num_desired_peaks).
    """

    height_threshold = np.percentile(y, 90)  # Height threshold fixed at 90th percentile

    for prominence_percentile in range(min_prominence_percentile, 101, 2):
        prominence_threshold = np.percentile(y, prominence_percentile)
        peaks, properties = find_peaks(y, height=height_threshold, prominence=prominence_threshold)

        if peaks.size >= num_desired_peaks:
            peak_prominences = properties.get("prominences", np.zeros(len(peaks)))
            sorted_peak_indices = np.argsort(peak_prominences)[::-1]
            top_peak_indices = sorted_peak_indices[:num_desired_peaks]
            top_peaks = peaks[top_peak_indices]
            break  # Exit loop once enough peaks are found
        elif prominence_percentile == 100 and peaks.size < num_desired_peaks:
            peak_prominences = properties.get("prominences", np.zeros(len(peaks)))
            sorted_peak_indices = np.argsort(peak_prominences)[::-1]
            top_peak_indices = sorted_peak_indices[:min(num_desired_peaks, len(peaks))]
            top_peaks = peaks[top_peak_indices]
            print("Could not reach desired number of peaks. Returning available peaks.")
            break
        elif prominence_percentile == 100 and peaks.size == 0: # Check if no peaks are found at 100%
            print("No peaks found even at 100% prominence.")
            top_peaks = np.array([])
            break

    feature_vector = []
    available_peaks = min(num_desired_peaks, len(top_peaks)) # Number of real peaks

    for i in range(available_peaks):
        if top_peaks.size > 0: # Check if there is at least one peak
            feature_vector.append(x[top_peaks[i]])

    # Pad with -1 if necessary:
    while len(feature_vector) < num_desired_peaks:
        feature_vector.append(-1)

    feature_vector = np.array([feature_vector])
    feature_vector = feature_vector.flatten().reshape(1, -1)
    return feature_vector, top_peaks

def bucketize(x, y, bucket_width=0.50):
    min_x = np.min(x)
    max_x = np.max(x)

    num_buckets = int(np.ceil((max_x - min_x) / bucket_width)) # Calculate number of buckets
    bucket_max_y = np.full(num_buckets, -np.inf) # Initialize with negative infinity

    current_bucket_start = np.floor(min_x / bucket_width) * bucket_width

    for i in range(num_buckets):
        current_bucket_end = current_bucket_start + bucket_width

        # Find indices of x-values within the current bucket
        indices_in_bucket = np.where((x >= current_bucket_start) & (x < current_bucket_end))[0]

        if indices_in_bucket.size > 0:  # Check if any points are in the bucket
            max_y_in_bucket = np.max(y[indices_in_bucket])
            bucket_max_y[i] = max_y_in_bucket

        current_bucket_start = current_bucket_end

    # Replace -inf with NaN for empty buckets (optional, but often useful)
    bucket_max_y = bucket_max_y/np.max(y)
    for i in range(len(bucket_max_y)):
        bucket_max_y[i] = bucket_max_y[i]**2
    
    bucket_max_y[bucket_max_y == -np.inf] = np.nan


    return np.array(bucket_max_y)



if __name__ == "__main__":
    datafolderpath = 'C:/Personal/Honors Thesis/src/Data2/'
    filename = '103695-050-6.dat'
   
    #<<<<<<<<<<<<<<<<< Laptop Path. Comment out if not on laptop >>>>>>>>>>>>>>
    #datafolderpath = 'C:/Users/jgao0/.vscode/Personal/Honors-Thesis/Data/'

    data = np.loadtxt(datafolderpath + filename, delimiter=None)
    x = data[:, 0]  # Angle values
    y = data[:, 1]  # Intensity values

    #print(bucketize(x,y))

    
    feature_vector, top_peaks = find_and_select_peaks(x,y)
    # Plot results
    plt.plot(x, y, label="XRPD Pattern")

    if top_peaks.size > 0:
        plt.plot(x[top_peaks], y[top_peaks], "ro", label=f"Top {len(top_peaks)} Peaks")

    plt.xlabel("2θ (degrees)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.show()

    # Print detected peak positions
    if top_peaks.size > 0:
        print("Detected top peaks at:", x[top_peaks])
    else:
        print("No peaks found above thresholds.")

    print("Feature Vector:", feature_vector)
    

    #TODO:
    #Fix padding for np.array feature vectors produced from bucketize. Try clustering