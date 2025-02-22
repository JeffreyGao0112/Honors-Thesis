import numpy as np
import os
import matplotlib.pyplot as plt

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

def plot_and_save_xy(folder_path, save_folder):
    """Plots x,y data from .dat files and saves the plots as images."""
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    xy_data_list = read_xy_files(folder_path)

    for x_data, y_data, filename in xy_data_list:
        plt.figure()  # Create a new figure for each plot
        plt.plot(x_data, y_data)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Plot of {filename}")
        save_path = os.path.join(save_folder, f"{os.path.splitext(filename)[0]}.png") #save as png
        plt.savefig(save_path)
        plt.close()  # Close the figure to free up memory

# Example Usage:
data_folder = "C:/Personal/Honors Thesis/src/Data2/"  # Replace with your data folder
save_folder = "C:/Personal/Honors Thesis/src/Plots/" # Replace with your save folder

plot_and_save_xy(data_folder, save_folder)