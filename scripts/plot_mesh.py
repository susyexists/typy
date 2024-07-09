# Import typy library
from typy import *
# Input files
file_path = f"./input/"
nscf = "nscf.out"
wout = "NbSe2.wout"
hr = "NbSe2_hr.dat"
# Define metallic band
metallic_band_index = 6
# Create NxN mesh
N = 100
# Create model
model = typy(file_path, nscf, wout, hr)
# Plot metallic band
plot_electron_mesh(model, N, metallic_band_index,
                   save="./output/band_mesh.png")
