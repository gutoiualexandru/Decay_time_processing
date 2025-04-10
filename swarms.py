import numpy as np
import torch
import pyswarms as ps
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


##############################################################################
# 1) DEFINE THE FIT FUNCTION: y = a * exp(-b / x) + c
##############################################################################
def fit_function(params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        params: shape (3,) or (batch_size,3) => [a, b, c]
        x:      shape (num_points,)
    Returns:
        y_pred: shape (num_points,) if a single param set
                or (batch_size, num_points) if multiple sets
    """
    x = x.float()
    params = params.float()

    if params.ndim == 1:
        # single set => add batch dimension
        params = params.unsqueeze(0)  # shape => (1,3)

    a = params[:, 0].unsqueeze(-1)  # shape => (batch_size,1)
    b = params[:, 1].unsqueeze(-1)
    c = params[:, 2].unsqueeze(-1)

    # Reshape x for broadcasting => (1,num_points)
    x = x.unsqueeze(0)

    # Avoid division by zero

    y_pred = a * torch.exp(-x / b) + c

    # If only one set of params, squeeze out the batch dimension
    if y_pred.shape[0] == 1:
        y_pred = y_pred.squeeze(0)

    return y_pred


##############################################################################
# 2) DEFINE COST FUNCTION FOR PSO (MSE)
##############################################################################
def mse_cost_function(params_array: np.ndarray,
                      x_data_t: torch.Tensor,
                      y_data_t: torch.Tensor) -> np.ndarray:
    """
    Params for PySwarms:
        params_array: shape (n_particles, 3) => each row is (a, b, c)
    Returns:
        costs: shape (n_particles,) => MSE for each particle
    """
    n_particles = params_array.shape[0]
    costs = np.zeros(n_particles, dtype=np.float32)

    for i in range(n_particles):
        p = torch.from_numpy(params_array[i])
        y_pred = fit_function(p, x_data_t)
        mse = torch.mean((y_pred - y_data_t)**2).item()
        costs[i] = mse

    return costs


##############################################################################
# 3) MAIN FITTING FUNCTION
##############################################################################
def fit_vector_with_pso(y_data: np.ndarray,
                        n_particles: int = 30,
                        iters: int = 100,
                        param_bounds=[],
                        plot: bool=True):
    """
    1) Takes a 1D vector y_data (already preprocessed if needed).
    2) Fits the model y = a * exp(-b / x) + c using Particle Swarm.
    3) Plots the data and the fitted curve.

    Returns:
        best_params (np.ndarray) => shape (3,) => [a, b, c]
        best_cost   (float)      => MSE
    """
    # A) Build x_data from indices
    index=cut(y_data)
    y_copy=np.copy(y_data)
    x_copy=np.arange(len(y_data), dtype=np.float32)
    y_data=y_data[index:]
    num_points = len(y_data)
    x_data = np.arange(num_points, dtype=np.float32)

    # Convert to torch tensors
    x_data_t = torch.from_numpy(x_data)
    y_data_t = torch.from_numpy(y_data.astype(np.float32))

    # B) Set up PySwarms
    options = {'c1': 1, 'c2': 1, 'w': 1}
    if len(param_bounds)>0:
        (lower_bounds, upper_bounds) = param_bounds
        optimizer = ps.single.GlobalBestPSO(
            n_particles=n_particles,
            dimensions=3,  # a, b, c
            options=options,
            bounds=(lower_bounds, upper_bounds)
        )
    else:
        optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=3,  # a, b, c
        options=options)

    cost_func = lambda p: mse_cost_function(p, x_data_t, y_data_t)

    # C) Optimize
    best_cost, best_params = optimizer.optimize(cost_func, iters=iters)

    print("Best-fit parameters [a, b, c]:", best_params)
    print("Best MSE found:", best_cost)

    # D) Plot the data and fitted curve
    # Make a dense x array for a smoother line
    x_fit = np.linspace(x_data.min(), x_data.max(), 300, dtype=np.float32)
    x_fit_t = torch.from_numpy(x_fit)
    best_params_t = torch.from_numpy(best_params)
    y_fit = fit_function(best_params_t, x_fit_t).detach().numpy()


    H=hessian(best_params_t[0],best_params_t[1],best_params_t[2], x_data, y_data)
    cov = invert_hessian(H)  # approximate covariance
    # If the Hessian is singular or near-singular, this might fail or have large values
    variances = np.diag(cov)
    # Negative or very large values can indicate a near-degenerate solution
    param_stds = np.sqrt(np.abs(variances))  # simple approach

    print("Approx. Covariance matrix:\n", cov)
    print("Estimated param std devs [sigma_a, sigma_b, sigma_c]:", param_stds)
    if plot:
        plt.figure()
        plt.scatter(x_copy, y_copy, label='Data', marker='o')
        plt.plot([index,index], [max(y_copy), min(y_copy)], color='r', linestyle=':')
        for i in range(len(x_fit)):
            x_fit[i]+=index
        plt.plot(x_fit, y_fit, color='r', label='Fitted Curve')
        plt.plot([x_fit[0], x_fit[len(x_fit)-1]],[best_params_t[0]+best_params_t[2]]*2, color='k', linestyle=':')
        plt.plot([x_fit[0], x_fit[len(x_fit)-1]],[best_params_t[2]]*2, color='k', linestyle=':')
        plt.xlabel("x (index)")
        plt.ylabel("y")
        plt.title("Fitting y = a * exp(-x / b) + c")
        # plt.xlim(min(x_copy), max(x_copy))
        # plt.ylim(min(y_copy), max(y_copy))
        plt.grid(True)
        plt.legend()
        plt.show()

    return best_params, best_cost, param_stds

#Hessian

def hessian(a, b, c, x_data, y_data):
    """
    Computes the 3x3 Hessian of the MSE cost for the model:
       f(a,b,c; x) = a * exp(-x/b) + c
    Data: (x_data[i], y_data[i])
    """

    N = len(x_data)
    H = np.zeros((3, 3), dtype=float)

    for i in range(N):
        # Convert to float explicitly (if they aren't already)
        x_i = float(x_data[i])
        y_i = float(y_data[i])

        # residual r_i
        r_i = float(y_i - (a * np.exp(-x_i/b) + c))

        # 1st derivatives
        dr_da = - np.exp(-x_i/b)                       # float
        dr_db = - a * np.exp(-x_i/b) * (x_i / b**2)     # float
        dr_dc = -1.0                                    # float

        # Pack them in a numeric array
        dr = np.array([dr_da, dr_db, dr_dc], dtype=float)

        # 2nd derivatives of r_i
        # d2r/da^2 = 0
        d2r_dadb = - (x_i / b**2) * np.exp(-x_i/b)    # derivative wrt both a & b
        # d2r/db^2:
        #   g(b) = a*(x_i/b^2) * exp(-x_i/b)
        #   dr_db = -g(b) => second derivative = -g'(b)
        g_prime = a * x_i * np.exp(-x_i/b) / (b**3) * ((x_i / b) - 2.0)
        d2r_db2 = - g_prime

        # build 3x3 second-derivative matrix for r_i
        d2r = np.zeros((3, 3), dtype=float)
        # da db
        d2r[0, 1] = d2r_dadb   # a,b
        d2r[1, 0] = d2r_dadb   # b,a
        # db^2
        d2r[1, 1] = d2r_db2

        # Outer product of first derivatives
        outer_dr = np.outer(dr, dr)  # shape (3,3), float

        # local Hessian from point i
        # cost = (1/N) sum of r_i^2 => partial derivative factor = 2/N
        local_H = outer_dr + r_i * d2r
        H += (2.0 / N) * local_H

    return H

def invert_hessian(H):
    """
    Attempt to invert the Hessian to get the approximate covariance matrix.
    """
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        # Hessian might be singular/ill-conditioned
        cov = np.full_like(H, np.nan)
    return cov

import pandas as pd


def read_nth_line(n, csv_file):
    """
    Returns the n-th row (0-based) of a CSV file as an np.ndarray of numeric data.
    
    Args:
        n (int): The row index (0-based) to return.
        csv_file (str): Path to the CSV file.
        
    Raises:
        IndexError: If the requested row index is out of bounds.
        
    Returns:
        numpy.ndarray: Numeric values from the n-th row.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Check bounds
    if n < 0 or n >= len(df):
        raise IndexError(f"Row index {n} is out of bounds for this CSV.")
        
    # Select the n-th row (Series)
    row_data = df.iloc[n]
    
    # Convert row data to numeric if necessary, ignoring non-numeric columns
    row_data_numeric = pd.to_numeric(row_data, errors='coerce').dropna()
    
    # Return as a numpy array
    return row_data_numeric.values

def determine_bounds(y):
    #(y-c)/a=e^(-t/tau)=>-t/tau=ln((y-c)/a)=>tau=-t/ln((y-c)/a)=t/ln(a/(y-c))
    #min(t)=1
    #max(t)=len(y)
    #(a/(y-c))=f*amp/(y_max-y_min)
    #min(tau)=1/ln(f*amp/(y_max-y_min)) f=1
    #max(tau)=len(y)/ln(f*amp/(y_max-y_min)) f=1

    y_min=min(y)
    y_max=max(y)
    amp=y_max-y_min
    a_bounds=np.array([amp*0.5, amp*100])
    c_bounds=np.array([y_min*0.9, y_min+amp/4])
    b_min=1
    b_max=len(y)
    b_bounds=np.array([b_min, b_max])
    bounds=np.array([[a_bounds[0], b_bounds[0], c_bounds[0]],[a_bounds[1], b_bounds[1], c_bounds[1]]])
    return bounds

def plot_histogram(v, bins=20, title="Histogram"):
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
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

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

##############################################################################
# 4) EXAMPLE USAGE (mock data & preprocessing)
##############################################################################

if __name__ == "__main__":

    # n_points = 100
    # x_temp = np.arange(0, n_points , 1, dtype=np.float32)  # [1..30]
    # true_a, true_b, true_c = 15, 50, 20
    # y_true = true_a * np.exp(-x_temp/true_b ) + true_c
    # noise = 0 * np.random.randn(len(x_temp))
    # y_noisy = y_true + noise

    # y_preprocessed = y_noisy  # or some transform
    # taus=[]
    # df = pd.read_csv("20250130/ringdowns04.csv")
    # L=len(df)
    for i in range(10):
        y=read_nth_line(i, "20250130/ringdowns04.csv")
        fit_params, fit_mse, uncertainties = fit_vector_with_pso(
            y_data=y,
            n_particles=1000,
            iters=100,
            param_bounds=determine_bounds(y),
            plot=True
        )
        # taus.append(fit_params[1]/200)
    
    # print(taus)
 #   taus=[281.58163467193276, 225.35373844372543, 268.0766135978745, 221.05420706679843, 263.2322805519707, 247.18814442612165, 229.11607729843126, 249.58081972022978, 255.81477883938038, 242.5456197366775, 217.89496058177792, 209.49785041994875, 237.3028963388242, 303.2501658656456, 213.19901639478343, 254.98222236627225, 253.9385670423435, 238.89219544223835, 256.92831340636394, 283.0608783104019, 235.44197755023703, 221.69399762270007, 266.7616341478931, 268.40126231916054, 227.21197270965956, 252.98290683211235, 314.3344065664832, 248.15691449866745, 223.77767646691285, 263.0511058727593, 220.67698314944383, 265.5493996865742, 206.86483486278132, 191.06759712866887, 229.83555130346537, 256.0266970826233, 195.09403575177683, 215.72680204906445, 216.39179706767254, 276.4640212271673, 231.28355523410812, 254.08936952100441, 228.28102415817193, 247.4567461869151, 223.55684876768385, 275.10270812612396, 251.55370560983428, 254.19561261319166, 251.04903571503166]

    # taus=[5.094335091384637, 6.055358037293488, 6.171249640811386, 5.537742487419546, 5.074749096201486, 5.870429678891348, 5.117170070864504, 5.371365169926608, 5.139326987403781, 5.596600161306606, 4.9497584853583705, 6.854214601071808, 6.14762708461762, 6.001174349440113, 5.692176177704132, 4.9106078985339625, 6.31915952230622, 6.439021983157418, 6.708999541838193, 5.599966319533378, 6.54839350562559, 5.3413248793658745, 5.370831405651843, 5.795002962474768, 5.001412417831412, 5.265927366778409, 5.832347294165429, 5.479001632684434, 5.770431493876108, 5.210824098489029, 5.108285852994716, 4.885307139797964, 5.856250994103339, 6.052008062593433, 5.62597981743932, 4.632725422573158, 6.490254304505106, 5.317223822671679, 5.468360648196709, 5.338174533607271, 6.461593811197572, 5.154961250705067, 5.929710618831741, 5.388815086095811, 6.185657887064606, 6.73061059187763, 5.106855251932829, 5.544349841690479, 5.608700714126335]
    # for i in range(len(taus)):
    #     taus[i]=taus[i]/2
    # plot_histogram(taus)
    # y=read_nth_line(10, "20250130/ringdowns04.csv")

    # # y=y[250:]
    # # # y=y_noisy
    # fit_params, fit_mse, uncertainties = fit_vector_with_pso(
    #     y_data=y,
    #     n_particles=1000,
    #     iters=100,
    #     param_bounds=determine_bounds(y),
    #     plot=True
    # )
