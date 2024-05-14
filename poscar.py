#Tightbinding Library
import src as typy

file_path = f"./bulk/" #Location of data files
poscar = "CONTCAR" #name of poscar file
hr = "wannier90_hr.dat"#name of wannier90 hr file
ef = 4.1376 #fermi energy
n_points = 4 #number of q points
k_mesh = [1,1,1] #k mesh size
symmetry_points = [[0.0,0.0,0.0],[0,0.5,0.0],[1/3,1/3,0.0],[0,0,0]] #symmetry point coordinates
labels=['Γ','M','K','Γ'] # symmetry point labels

#Create tightbinding model
model = typy.model(path=file_path,poscar=poscar,hr=hr,ef=ef)
sym,path = typy.path_create(n_points,symmetry_points)
mesh = typy.mesh_cartesian(k_mesh)
mesh_energy = model.calculate_energy(mesh)
suscep_bands = typy.fermi_bands(mesh_energy)
suscep = model.suscep_path(path,mesh,suscep_bands)
typy.plot_susceptibility(suscep,sym,labels,save=True)
typy.write_susceptibility(path,sym,suscep)