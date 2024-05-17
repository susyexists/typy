# Numerical tools
from scipy.constants import physical_constants
# Matrix inversion
from numpy.linalg import inv
import numpy as np
# Plotting
import matplotlib.pyplot as plt
# Data analysis
import pandas as pd
# Parallel computation
from joblib import Parallel, delayed
import multiprocessing
# Physical constants
kb = physical_constants['Boltzmann constant in eV/K'][0]
g_vec = np.array([[0.86602505, 0.5], [0, 1]])
inverse_g = np.linalg.inv(g_vec)
import psutil
from . import utils
from tqdm import tqdm

class model:
    def __init__(self, hr, path="./",nscf=False,poscar=False, ef=0,read_ef=False,shift=0,num_core=False):
        if num_core!=False:
            self.num_cores= num_core
        else:
            self.num_cores = multiprocessing.cpu_count()
        self.shift = shift
        self.path = path
        if read_ef:
            self.fermi_energy = read_efermi(path+nscf)+self.shift
        else:
            self.fermi_energy=ef
        if nscf:
            self.g_vec = read_gvec(path+nscf)
        if poscar:
            lattice_vector = utils.read_poscar(path+poscar)
            self.g_vec = utils.crystal2reciprocal(lattice_vector)
        self.g_length = 1
        self.data = read_hr(path+hr)
        self.hopping = self.data[0]
        self.nbnd = int(np.sqrt(len(self.data[0])/len(self.data[2])))
        self.points = len(self.data[2])
        self.sym = self.data[2]
        self.h = self.hopping.reshape(self.points, self.nbnd*self.nbnd)
        self.x = self.data[1].reshape(3, self.points, self.nbnd*self.nbnd)

    def fourier(self, k):
        kx = np.tensordot(k, self.x, axes=(0, 0))
        transform = np.dot(self.sym, np.exp(-1j*self.super_cell*kx)
                           * self.h).reshape(self.nbnd, self.nbnd)
        return(transform)

    def eig(self, k):
        val = []
        vec = []
        for i in range(len(k)):
            sol = np.linalg.eigh(self.fourier(k[i]))
            val.append(sol[0])
            vec.append(sol[1])
        return (val, vec)

    def solver(self, k):
        kx = np.tensordot(k, self.x, axes=(0, 0))
        transform = np.dot(self.sym, np.exp(-1j*kx*2*np.pi)
                           * self.h).reshape(self.nbnd, self.nbnd)
        val, vec = np.linalg.eigh(transform)
        return(val)

    def calculate_energy(self, path, band_index=False):
        path = path
        results = Parallel(n_jobs=self.num_cores)(
            delayed(self.solver)(i) for i in path)
        res = np.array(results).T-self.fermi_energy
        if band_index==False:
            return (res)
        else:
            return (res[band_index])

    def suscep(self, point, mesh, mesh_energy, mesh_fermi, bands,T=1,delta=0.0000001,fermi_shift =0 ):
        real= 0
        imag = 0
        shifted_energy = self.calculate_energy(point+mesh)
        shifted_fermi = fd(shifted_energy,T)
        for i in bands:
            for j in bands:
                num = mesh_fermi[i]-shifted_fermi[j]
                den = mesh_energy[i]-shifted_energy[j]+1j*delta
                real += np.average(num/den)
                imag += np.average(delta_function(mesh_energy[i])*delta_function(shifted_energy[j]))
        return([-real.real,imag])
    
    def suscep_path(self,q_path,k_mesh,band_index,T=1):
        en_k = self.calculate_energy(k_mesh)
        fd_k = fd(en_k,T)
        res = [self.suscep(point=q,mesh= k_mesh,mesh_energy=en_k,mesh_fermi = fd_k,bands=band_index) for q in tqdm(q_path)]
        return np.array(res).T

    def plot_electron_path(self, band, sym, labels, ylim=None, save=None, temp=None):
        # Plot band
        plt.figure(figsize=(7, 6))
        for i in band:
            plt.plot(i, c="blue")
        plt.xticks(ticks=sym, labels=labels, fontsize=15)
        plt.xlim(sym[0], sym[-1])
        for i in sym[1:-1]:
            plt.axvline(i, c="black", linestyle="--")
        plt.axhline(0, linestyle="--", color="red")
        if ylim == None:
            plt.ylim(-0.6, 0.8)
        else:
            plt.ylim(ylim)
        if temp != None:
            plt.title(f"σ = {temp}", fontsize=15)
        if self.shift != 0:
            plt.title(
                r"$\delta \epsilon_{Fermi} = $"f" {self.shift} eV", fontsize=15)
        plt.ylabel("Energy (eV)", fontsize=15)
        if save != None:
            plt.savefig(save)

            
def mesh_cartesian(num_points=[6,6,6], factor=1):
    x = np.linspace(0, 1, num_points[0])
    y = np.linspace(0, 1, num_points[1])
    z = np.linspace(0, 1, num_points[2])
    three_dim = np.array([[i, j,k] for i in x for j in y for k in z])
    return (three_dim*factor)

def mesh_crystal(N):
    mesh = mesh_cartesian(N)
    t_mesh = np.dot(g_vec.T, mesh.T)
    return t_mesh


def GMKG(num_points, g_length=1):
    mult = 0.75*num_points/1.00786
    GM = int(np.linalg.norm(
        np.array([[0, 0.5]])-np.array([[0.0001, 0.0001]]))*mult)
    MK = int(
        np.linalg.norm(np.array([[0.33333, 0.333333]])-np.array([[0, 0.5]]))*mult)
    KG = int(np.linalg.norm(np.array([[0.0001, 0.0001]]) -
             np.array([[0.33333, 0.333333]]))*mult)
    path1 = np.array([np.zeros(GM), np.linspace(0.0001, g_length/2, GM)]).T
    path2 = np.array([np.linspace(0, g_length/3, MK), -1/2.1 *
                      np.linspace(0, g_length/3, MK)+g_length/2]).T
    path3 = np.array([np.linspace(g_length/3, 0.0001, KG),
                      np.linspace(g_length/3, 0.0001, KG)]).T
    path = np.concatenate([path1, path2, path3])
    # print("Length of the path is ", len(path))
    sym = [0, GM, GM+MK, GM+MK+KG]
    label = ['Γ', 'M', 'K', 'Γ']
    return path, sym, label


def Symmetries(fstring):
    f = open(fstring, 'r')
    x = np.zeros(0)
    for i in f:
        x = np.append(x, float(i.split()[-1]))
    f.close()
    return x


def plot_fs(band, fs_thickness=0.01, title=None):
    # Imaging cross sections of fermi surface using a single calculation
    df = pd.DataFrame()
    x,y = t_mesh(int(np.sqrt(len(band))))
    df['x'] = x
    df['y'] = y
    df['E'] = band
    fs = df.query(f' {-fs_thickness} <= E <= {fs_thickness}')
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(fs.x, fs.y)
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)
    if title != None:
        plt.title(title, fontsize=15)
    plt.show()


def fd(E,T=1):
    E=E.astype(dtype=np.float128)
    return 1/(1+np.exp(E/(kb*T)))

def delta_function(x, epsilon=0.00001):
    return (1 / np.pi) * epsilon / (x ** 2 + epsilon ** 2)


def read_gvec(path):
    lines = open(path, 'r').readlines()
    g_vec = np.zeros(shape=(3, 3))
    count = 0
    for i in lines:
        if "b(" in i:
            if count == 3:
                continue
            else:
                g_vec[count] = i.split()[3:6]
                count += 1
    g_vec = g_vec.T[:2].T[:2]
    return (g_vec)


def read_hr(path):
    lines = open(path, 'r').readlines()
    sym_line = int(np.ceil(float(lines[2].split()[0])/15))+3
    sym = np.array([int(lines[i].split()[j]) for i in range(3, sym_line)
                    for j in range(len(lines[i].split()))])
    hr_temp = np.array([float(lines[i].split()[j]) for i in range(
        sym_line, len(lines)) for j in range(len(lines[i].split()))])
    hr = hr_temp.reshape(-1, 7).T
    x = hr[0:3]
    hopping = hr[5]+1j*hr[6]
    return (hopping, x, sym)


def read_efermi(path):
    lines = open(path, 'r').readlines()
    e_fermi = 0
    for i in lines:
        if "the Fermi energy is" in i:
            e_fermi = float(i.split()[-2])
            return e_fermi

def plot_electron_mesh(band, N, metallic_band_index, xlim, ylim, plot_factor=5, save=None, temp=None, cmap='jet'):
    x, y = t_mesh(N)
    plt.figure(figsize=(plot_factor*xlim+1, plot_factor*ylim))
    plt.scatter(x, y, c=band[metallic_band_index], cmap=cmap)
    plt.colorbar()
    plt.xlim(-xlim, xlim)
    plt.ylim(-ylim, ylim)
    # plt.axis("equal")
    plt.xticks([])
    plt.yticks([])
    if temp == None:
        plt.title("")
    else:
        plt.title(f"σ = {temp}")
    if save != None:
        plt.savefig(save)


def density_of_states(energy, band_index=False, dE=1e-2):
    if band_index:
        E = energy[band_index]
    else:
        E=energy
    # Initial empty array for dos
    dos = np.zeros(len(E))
    # Iterate over each energy
    for i in range(len(E)):
        # Delta function approxiation for given value of energy over all states
        delta_array = np.where(abs(E[i]-E) < dE, np.ones(len(E)), 0)
        delta_average = np.average(delta_array)
        dos[i] = delta_average
    return dos

def ram_check():
    # Getting % usage of virtual_memory ( 3rd field)
    print('RAM memory % used:', psutil.virtual_memory()[2])
    # Getting usage of virtual_memory in GB ( 4th field)
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    
def rotate(vector,angle):
    matrix = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])    
    transform = np.dot(matrix,vector.T)
    return(transform)

def triangle_mesh(N):
    x,y = mesh_cartesian(N).T
    df = pd.DataFrame()
    df['x']=x
    df['y']=y
    triangle = df.query("y<=sqrt(3)*x").query("y<=-sqrt(3)*x+sqrt(3)").values
    return triangle.T

def hexagon_cartesian(N):
    triangle = triangle_mesh(N).T
    fold =6 
    grid = np.zeros(shape=(fold,len(triangle),2))
    for n in range(0,fold):
        theta = n*np.pi/3
        # rx, ry =rotate(triangle, theta)
        grid[n]=rotate(triangle, theta).T
        # print(len(rx))
        # plt.scatter(rx,ry)
    grid = grid.reshape(-1,2).T
    grid = grid/max(grid[1])/2
    return grid
    
def hexagon_crystal(N,g_vec=g_vec):
    grid = np.dot(hexagon_cartesian(N).T,inverse_g)
    return grid

def cartesian2crystal(cartesian):
    grid = np.dot(g_vec,cartesian)
    return grid

def untangle(band,cross):
    fix = band.copy()
    num_cross = len(cross)
    for i in range(num_cross):
        temp_1 = fix[cross[i][0]].copy()
        temp_2 = fix[cross[i][1]].copy()
        point = cross[i][2]
        fix[cross[i][1]][:point] = temp_1[:point]
        fix[cross[i][0]][:point] = temp_2[:point]
    return fix

def ph_cross(ph,tolerance,offset=50):
    ph_xs=[]
    pointer = 0
    for q in range(offset,ph.shape[1]-offset):
        for i in range(ph.shape[0]):
            for j in range(ph.shape[0]):
                if i<j:
                    if abs(ph[i][q]-ph[j][q])<tolerance:
                        if(q!= pointer+1):
                            # print(i,j,q)
                            ph_xs.append([i,j,q])
                        pointer = q          
    return ph_xs
    
def g_cross(g,tolerance,offset=50):
    g_xs=[]
    pointer = 0
    for q in range(offset,g.shape[1]-offset):
        for i in range(g.shape[0]):
            for j in range(g.shape[0]):
                if i<j:
                    if abs(g[i][q]-g[i][q-1])>tolerance:
                        if abs(g[j][q]-g[j][q-1])>tolerance:
                                # print(i,j,q)
                                g_xs.append([i,j,q])
    return g_xs
    

def find_cross(band,parameter):
    xs=[]
    point_pair=[]
    for i in range(len(band)):
        grad = np.gradient(band[i])
        for j in range(1,len(band[i])):
            if abs(grad[j]-grad[j-1])>parameter:
                # print(i,j)
                point_pair.append([i,j])
    # print(point_pair)
    for i in point_pair:
        point=i[1]
        begin=i[0]
        for j in point_pair:
            if i[0]!=j[0]:
                if i[1]==j[1]:
                    end=j[0]
                    xs.append([begin,end,point])
    xs_sort = xs[xs[:, 2].argsort()]    
    return np.array(xs_sort)

