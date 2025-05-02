# Ringdown Code By Alexandru

## Usage
### Process Data
Use the `process_data.py`, which takes a csv file as input, does the analysis and spits out a `npy` file as the output containing your fit information.
### Read
The `read.py` file reads the analysis results (modify the name of the analysis file if needed) and plots out the relevant statistics (error, std etc). One can always write a new version of read to plot out whatever you want.
### Swarms
All the fitting functions relevant for fitting and calculations.
### Viz
For testing and checking a signal, plots the moving average as well.

## Specification
### Cropping
In the ringdown signal, the moving average method finds the highest and lowest point. Then from there, the code takes the point at which the signal is 5% lower than the highest point. Your data should not include a part where the signal comes back up. For instance, if light turns back on and the cavity starts accumulating light again, this is not taken into account in the code and your data recording should be fixed to accomodate this fact.
