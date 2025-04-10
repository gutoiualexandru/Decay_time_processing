from swarms import *
import pandas as pd
import numpy as np
# Import your custom functions from swarms.py
# from swarms import read_nth_line, cut, fit_vector_with_pso, determine_bounds

def process_and_save(csv_file, output_npy="results.npy"):
    df = pd.read_csv(csv_file)
    L = len(df)

    # This will hold one dictionary per row
    results = []

    for i in range(L):
        # 1) Extract the i-th row as a numeric vector y
        y = read_nth_line(i, csv_file)  # from your swarms.py
        
        # 2) Find the cut point
        cut_point = cut(y)
        
        # 3) Fit with PSO
        fit_params, fit_mse, uncertainties = fit_vector_with_pso(
            y_data=y,
            n_particles=1000,
            iters=100,
            param_bounds=determine_bounds(y),
            plot=False
        )

        # 4) Prepare a dictionary for this row
        row_dict = {
            "y": y,  # the original vector
            "cut_point": cut_point,
            "fit_params": fit_params,
            "fit_mse": fit_mse,
            "uncertainties": uncertainties
        }
        
        # 5) Append to our list of results
        results.append(row_dict)

    # 6) Save the entire list to a .npy file
    #    We need allow_pickle=True because we're saving Python objects (dicts, arrays)
    np.save(output_npy, results, allow_pickle=True)
    print(f"Saved {len(results)} results to {output_npy}")

# Example usage:

if __name__ == "__main__":
    file_name = "20250130/ringdowns04.csv"
    output_npy = "analysis_results.npy"
    process_and_save(file_name, output_npy)