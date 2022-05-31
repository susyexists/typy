# Numerical tools
from scipy.constants import physical_constants
import numpy as np
# Plotting
import matplotlib.pyplot as plt
# Data analysis
import pandas as pd
# Parallel computation
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()
# Physical constants
kb = physical_constants['Boltzmann constant in eV/K'][0]


class typy:
    def __init__(self, path, nscf, wout, hr, shift=0):
        self.shift = shift
        self.path = path
        self.fermi_energy = read_efermi(path+nscf)+self.shift
        self.g_vec = read_gvec(path+wout)
        self.g_length = 1
        self.data = read_hr(path+hr)
        self.hopping = self.data[0]
        self.nbnd = int(np.sqrt(len(self.data[0])/len(self.data[2])))
        self.points = len(self.data[2])
        self.sym = self.data[2]
        self.h = self.hopping.reshape(self.points, self.nbnd*self.nbnd)
        self.x = self.data[1].reshape(2, self.points, self.nbnd*self.nbnd)

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

    def parallel_solver(self, path):
        results = Parallel(n_jobs=num_cores)(
            delayed(self.solver)(i) for i in path)
        res = np.array(results).T-self.fermi_energy
        return (res)

    def suscep(self, point, mesh, mesh_energy, mesh_fermi, delta=0.0000001):
        shifted_energy = self.parallel_solver(point+mesh)[6]
        shifted_fermi = self.fermi(shifted_energy)
        num = mesh_fermi-shifted_fermi
        den = mesh_energy-shifted_energy+1j*delta
        res = -np.average(num/den)
        return(res)

    def plot_electron_path(self, band, sym, label, ylim=None, save=None, temp=None):
        # Plot band
        plt.figure(figsize=(7, 6))
        for i in band:
            plt.plot(i, c="blue")
        plt.xticks(ticks=sym, labels=label, fontsize=15)
        plt.xlim(sym[0], len(band.T))
        plt.axvline(sym[1], c="black", linestyle="--")
        plt.axvline(sym[2], c="black", linestyle="--")
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

    def t_mesh(self, N):
        mesh = mesh_2d(N)
        t_mesh = np.dot(self.g_vec.T, mesh.T)
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


def plot_fs(t_mesh, band, fs_thickness=0.01, title=None):
    # Imaging cross sections of fermi surface using a single calculation
    df = pd.DataFrame()
    df['x'] = t_mesh[0]
    df['y'] = t_mesh[1]
    df['E'] = band
    fs = df.query(f' {-fs_thickness} <= E <= {fs_thickness}')
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(fs.x, fs.y)
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)
    if title != None:
        plt.title(title, fontsize=15)
    plt.show()


def mesh_2d(N, factor=2):
    one_dim = np.linspace(-1, 1, N)
    two_dim = np.array([[i, j] for i in one_dim for j in one_dim])
    return (two_dim*factor)


def fermi(E, T=1):
    return 1/(1+np.exp(E/(kb*T)))


def read_gvec(path):
    lines = open(path, 'r').readlines()
    g_vec = np.zeros(shape=(3, 3))
    count = 0
    for i in lines:
        if "b_" in i:
            if count == 3:
                continue
            else:
                g_vec[count] = i.split()[1:]
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
    x = hr[0:2]
    hopping = hr[5]+1j*hr[6]
    return (hopping, x, sym)


def read_efermi(path):
    lines = open(path, 'r').readlines()
    e_fermi = 0
    for i in lines:
        if "the Fermi energy is" in i:
            e_fermi = float(i.split()[-2])
            return e_fermi


def plot_electron_mesh(model, N, metallic_band_index, save=None, temp=None):
    mesh = mesh_2d(N, factor=1.2)
    band = model.parallel_solver(mesh)[metallic_band_index]
    t_mesh = model.t_mesh(N)
    plt.figure(figsize=(6, 5))
    plt.scatter(t_mesh[0], t_mesh[1], c=band, cmap='jet')
    plt.colorbar()
    plt.xlim(-2.9, 2.9)
    plt.ylim(-2.6, 2.6)
    # plt.axis("equal")
    plt.xticks([])
    plt.yticks([])
    if temp == None:
        plt.title("")
    else:
        plt.title(f"σ = {temp}")
    if save != None:
        plt.savefig(save)
