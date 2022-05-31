# Import typy library
from typy import *
# Input files
file_path = f"./input/"
nscf = "nscf.out"
wout = "NbSe2.wout"
hr = "NbSe2_hr.dat"
# Create NxN mesh
mesh = mesh_2d(100)
# Create model
model = typy(file_path, nscf, wout, hr)
# Calculate dispersion
band = model.parallel_solver(mesh)
model.plot_electron_mesh(mesh, band=band_2d[6])
