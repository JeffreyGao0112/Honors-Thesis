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
            top_peak_prominences = peak_prominences[top_peak_indices]
            break  # Exit loop once enough peaks are found
        elif prominence_percentile == 100 and peaks.size < num_desired_peaks:
            peak_prominences = properties.get("prominences", np.zeros(len(peaks)))
            sorted_peak_indices = np.argsort(peak_prominences)[::-1]
            top_peak_indices = sorted_peak_indices[:min(num_desired_peaks, len(peaks))]
            top_peaks = peaks[top_peak_indices]
            top_peak_prominences = peak_prominences[top_peak_indices]
            print("Could not reach desired number of peaks. Returning available peaks.")
            break
        elif prominence_percentile == 100 and peaks.size == 0: # Check if no peaks are found at 100%
            print("No peaks found even at 100% prominence.")
            top_peaks = np.array([])
            top_peak_prominences = np.array([])
            break


    feature_vector = []
    for i in range(num_desired_peaks):
        if top_peaks.size > 0 and i < len(top_peaks):
            feature_vector.extend([x[top_peaks[i]], y[top_peaks[i]]])
        else:
            feature_vector.extend([-1, -1])  # Pad with -1 if fewer peaks

    return top_peaks, top_peak_prominences, feature_vector


if __name__ == "__main__":
    datafolderpath = 'C:/Personal/Honors Thesis/src/Data'
    filename = 'JJG1-00314-1.dat'
    # Load the data (replace this with actual file reading if needed)
    data = np.loadtxt(datafolderpath + filename, delimiter=None)
    x = data[:, 0]  # Angle values
    y = data[:, 1]  # Intensity values

    top_peaks, top_peak_prominences, feature_vector = find_and_select_peaks(x,y)
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