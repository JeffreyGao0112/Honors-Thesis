import numpy as np
import os
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def read_xy_data(file_path):
    """Reads x,y data from a single .dat file."""
    try:
        xy_data = np.loadtxt(file_path, comments="#")
        x_data = xy_data[:, 0]
        y_data = xy_data[:, 1]
        return x_data, y_data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None, None

def visualize_peaks_percentile(file_path, prominence, height_percentile):
    """
    Reads x,y data from a file, finds peaks, filters by height percentile, and visualizes them.

    Args:
        file_path: The path to the .dat file.
        prominence: The minimum prominence of peaks to find.
        height_percentile: The percentile of peak heights to use as a threshold.
    """
    x_data, y_data = read_xy_data(file_path)

    if x_data is None or y_data is None:
        return

    peaks, properties = find_peaks(y_data, prominence=prominence, height=0) #height=0 to allow for percentile filter later

    if not peaks.size: #if no peaks are found.
        print("No peaks found with the given prominence.")
        return

    peak_heights = properties["peak_heights"]
    height_threshold = np.percentile(peak_heights, height_percentile)

    filtered_peaks = peaks[peak_heights >= height_threshold]

    plt.figure()
    plt.plot(x_data, y_data)
    plt.plot(x_data[filtered_peaks], y_data[filtered_peaks], "x", color='red')

    plt.title(f"Peaks in {os.path.basename(file_path)} (Prominence={prominence}, Height Percentile={height_percentile})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# Example Usage:
file_path = "C:/Personal/Honors Thesis/src/Data2/103695-050-4a.dat"  # Replace with your file path
peak_prominence = 80  # Adjust prominence as needed
height_percentile = 10  # Adjust percentile as needed (e.g., 75 for top 25%)

visualize_peaks_percentile(file_path, peak_prominence, height_percentile)