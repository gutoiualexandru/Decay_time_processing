import numpy as np
import matplotlib.pyplot as plt

def plot_entry_result(entry, timestamp):
    """
    Plots the data and fit stored in an entry dictionary.
    
    The entry must contain:
      - "y": the original y vector (numpy array)
      - "cut_point": an integer indicating where the fitting/truncated data begins
      - "fit_params": a 3-element vector [a, b, c] fitted to the truncated data
         for the model y = a * exp(-x / b) + c.
    
    The resulting plot:
      - Shows the original data (scatter plot with x = index)
      - A vertical red dotted line at the cut point.
      - A red line representing the fitted curve on the truncated data.
      - Two horizontal black dotted lines at y = c and y = a + c.
    """
    # Unpack the entry dictionary
    y_data_full = entry["y"]           # Entire data vector
    cut_point = int(entry["cut_point"])  # where truncated fit starts
    fit_params = entry["fit_params"]   # [a, b, c]
    
    # Create x-axis for the full data
    x_full = np.arange(len(y_data_full), dtype=np.float32)
    
    # Truncated data to fit (from cut_point onward)
    y_truncated = y_data_full[cut_point:]
    num_points_trunc = len(y_truncated)
    x_trunc = np.arange(num_points_trunc, dtype=np.float32)
    
    # Create a dense x-axis for the fitted curve on the truncated domain
    x_fit_dense = np.linspace(x_trunc.min(), x_trunc.max(), 300, dtype=np.float32)
    # Shift x_fit_dense so that it aligns with original indices
    x_fit = x_fit_dense + cut_point
    
    # Unpack fit parameters
    a, b, c = fit_params
    
    # Evaluate the model on the dense x-axis, using truncated x coordinates.
    # (The model is defined on the truncated domain so that x=0 corresponds to cut_point.)
    y_fit = a * np.exp(-x_fit_dense / b) + c
    
    # Build the plot.
    plt.figure()
    
    # Plot the original data with markers.
    x_scaled=[i*timestamp for i in x_full]
    plt.scatter(x_scaled, y_data_full, label='Data', marker='o')
    cut_point=cut_point*timestamp
    # Plot a vertical line at the cut point.
    plt.plot([cut_point, cut_point], [np.max(y_data_full), np.min(y_data_full)],
             color='r', linestyle=':')
    
    # Plot the fitted curve, shifted to original coordinates.
    x_fit_scaled=[i*timestamp for i in x_fit]
    plt.plot(x_fit_scaled, y_fit, color='r', label='Fitted Curve')
    
    # Plot horizontal lines at y = c and y = a + c (per the original function).
    plt.plot([x_fit_scaled[0], x_fit_scaled[-1]], [c, c], color='k', linestyle=':')
    plt.plot([x_fit_scaled[0], x_fit_scaled[-1]], [a + c, a + c], color='k', linestyle=':')
    
    # Format the figure.
    plt.xlabel(r"Time ($\mu s)$")
    plt.ylabel("Amplitude (A.U.)")
    # plt.title("Fitting y = a * exp(-x / b) + c")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_histogram(v, bins=10, title="Histogram"):
    """
    Plots a histogram of the vector `v`, with mean and std in the legend.

    Args:
        v (array-like): Input data vector.
        bins (int): Number of histogram bins (default: 20).
        title (str): Title of the plot.
    """
    v = np.array(v)
    mean = np.mean(v)
    std = np.std(v)

    # Plot histogram
    plt.figure()
    counts, bins, patches = plt.hist(v, bins=bins, alpha=0.7, label=f"mean = {mean:.2f}\nstd = {std:.2f}")
    plt.xlabel(r"Measured Decay Time ($\mu s$)")
    # plt.ylabel("Frequency")
    # plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


loaded_results = np.load("analysis_results.npy", allow_pickle=True)
timestamp=10/2000 #in us
# Each element is a dict
first_result = loaded_results[18]
plot_entry_result(first_result, timestamp)

taus=[]
for i in range(len(loaded_results)):
    tau=loaded_results[i]['fit_params'][1]
    tau*=timestamp
    taus.append(tau)
plot_histogram(taus)

# Convert to NumPy array for convenience (if it's not already)
taus = np.array(taus)

# Compute mean and standard deviation
mean_t = np.mean(taus)
std_t = np.std(taus)

# Create the x-axis (indices)
x = np.arange(len(taus))

# Create the plot
plt.figure(figsize=(8, 5))
plt.scatter(x, taus, color='blue')
plt.axhline(mean_t, color='red', label=f'Mean = {mean_t:.2f}')
plt.axhline(mean_t + std_t, linestyle=':', color='black', label=f'Mean + STD = {mean_t + std_t:.2f}')
plt.axhline(mean_t - std_t, linestyle=':', color='black', label=f'Mean - STD = {mean_t - std_t:.2f}')

# Label axes and add grid and legend
plt.xlabel('Measurement')
plt.ylabel('Time (µs)')
# plt.title('Time Measurements Scatter Plot')
plt.grid(True)
plt.legend()

# Display the plot
plt.show()