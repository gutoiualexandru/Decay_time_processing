import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np

def cut(v):
    y_smoothed = savgol_filter(v, window_length=100, polyorder=1)
    m=max(y_smoothed)
    n=min(y_smoothed)
    amp=m-n
    i_min = np.argmin(v)  # Index of the minimum value
    i_max = np.argmax(v)  # Index of the maximum value
    # print(i_max, i_min)
    i=i_max
    while y_smoothed[i]>m-0.05*amp:
        i+=1
    print(i)
    return i
        
def plot_nth_line(n, csv_file):
    """
    Plots the n-th row (0-based) of a CSV file.
    
    Args:
        n (int): The row index (0-based) to plot.
        csv_file (str): Path to the CSV file.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Select the n-th row (Series)
    if n < 0 or n >= len(df):
        raise IndexError(f"Row index {n} is out of bounds for this CSV.")
        
    row_data = df.iloc[n]
    y_smoothed = savgol_filter(row_data, window_length=100, polyorder=1)
    p=cut(y_smoothed)
    # i_peak = np.argmax(y_smoothed)

    # Convert row data to numeric if necessary, ignoring non-numeric columns
    # If your data has all numeric columns, you can skip this step
    row_data_numeric = pd.to_numeric(row_data, errors='coerce').dropna()
    
    # Plot the row data
    plt.figure()
    plt.plot(range(len(row_data_numeric)), row_data_numeric, marker='o')
    plt.plot(range(len(y_smoothed)), y_smoothed, 'k')
    # plt.plot([p,p], [max(row_data_numeric), min(row_data_numeric)], color='r', linestyle=':')
    # plt.title(f'Row {n} of {csv_file}')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

def plot_nth_column(n, csv_file):
    """
    Plots the n-th column (0-based) of a CSV file.
    
    Args:
        n (int): The column index (0-based) to plot.
        csv_file (str): Path to the CSV file.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Check column index
    if n < 0 or n >= df.shape[1]:
        raise IndexError(f"Column index {n} is out of bounds for this CSV.")
        
    # Select the n-th column
    column_data = df.iloc[:, n]
    
    # Convert column data to numeric if necessary
    column_data_numeric = pd.to_numeric(column_data, errors='coerce').dropna()
    
    # Plot the column data
    plt.figure()
    plt.plot(range(len(column_data_numeric)), column_data_numeric, marker='x')
    # plt.title(f'Column {n} of {csv_file}')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

if __name__=='__main__':
    plot_nth_line(3, "20250130/ringdowns04.csv")      # Plots the 3rd row of "data.csv"

