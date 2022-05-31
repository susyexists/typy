#Tightbinding Library
from TB import *
#Location of data files
file_path = f"./data/"
nscf="nscf.out"
wout="NbSe2.wout"
hr="NbSe2_hr.dat"
T=0.001
metallic_band_index = 6
#Create q-path and k-mesh
q_points = 1000
k_mesh = 300
path, sym, label= GMKG(q_points)
mesh = mesh_2d(k_mesh)
#Define shift parameters
shift_search = linspace(-0.15,0.15,100)
#Generate susceptibility array
suscep = zeros(shape=(len(shift_search),len(path)))
#Calculate and print susceptibility
for i in range(len(suscep)):
    model = TB(file_path,nscf, wout,hr,shift=shift_search[i])
    mesh_energy = model.parallel_solver(mesh)[metallic_band_index]
    mesh_fermi = model.fermi(mesh_energy)
    sus_mesh = [ model.suscep(q, mesh,mesh_energy,mesh_fermi) for q in path ]
    suscep[i]=array(sus_mesh)
    savetxt(f"./export/sus_{shift_search[i]}.dat",suscep[i])