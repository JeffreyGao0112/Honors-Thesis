import numpy as np
import matplotlib.pyplot as plt

# Load data from the .dat file
# Replace 'your_file.dat' with the path to your .dat file
datafolderpath = 'C:/Personal/Honors Thesis/Data/'
filename = 'JJG1-00314-1.dat'

# Assuming the file is space or tab-delimited
try:
    data = np.loadtxt(datafolderpath + filename, delimiter=None)  # Automatically detects delimiters like space or tab
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Check if the file contains at least two columns
if data.shape[1] < 2:
    print("The file does not contain enough columns for x and y data.")
    exit()

# Extract x and y columns
x = data[:, 0]  # First column
y = data[:, 1]  # Second column

# Plot the data
plt.figure(figsize=(8, 6))
plt.plot(x, y, linestyle='-', color='blue', label='Connecting Line')
plt.scatter(x, y, s=20, color='red', label='Data Points', alpha=0.8, edgecolors='k')
plt.title('Plot of x vs y', fontsize=16)
plt.xlabel('x-axis', fontsize=14)
plt.ylabel('y-axis', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()

# Show the plot
plt.show()