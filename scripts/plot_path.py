# Import typy library
from typy import *
# Input files
file_path = f"./input/"
nscf = "nscf.out"
wout = "NbSe2.wout"
hr = "NbSe2_hr.dat"
# Number of k points
k_points = 1000
# Energy axis limits
ylim = [-2, 2]
# Generate path along GMKG points
path, sym, label = GMKG(k_points)
# Create model
model = typy(file_path, nscf, wout, hr)
# Calculate dispersion
band = model.parallel_solver(path)
# Plot bands
model.plot_electron_path(
    band, sym, label, ylim, save="./output/band_path.png")
