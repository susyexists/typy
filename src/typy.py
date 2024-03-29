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
num_cores = multiprocessing.cpu_count()
# Physical constants
kb = physical_constants['Boltzmann constant in eV/K'][0]
g_vec = np.array([[0.86602505, 0.5], [0, 1]])
inverse_g = np.linalg.inv(g_vec)
import psutil


class model:
    def __init__(self, path, nscf, hr, shift=0):
        self.shift = shift
        self.path = path
        self.fermi_energy = read_efermi(path+nscf)+self.shift
        self.g_vec = read_gvec(path+nscf)
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

    def calculate_energy(self, path, band_index=False,shift=0):
        path = path+shift
        results = Parallel(n_jobs=num_cores)(
            delayed(self.solver)(i) for i in path)
        res = np.array(results).T-self.fermi_energy
        if band_index==False:
            return (res)
        else:
            return (res[band_index])

    def suscep(self, point, mesh, mesh_energy, mesh_fermi, delta=0.0001):
        shifted_energy = self.calculate_energy(point+mesh)
        shifted_fermi = fd(shifted_energy)
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
            
def mesh_cartesian(N, factor=1):
    one_dim = np.linspace(0, 1, N)
    two_dim = np.array([[i, j] for i in one_dim for j in one_dim])
    return (two_dim*factor)

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


def fd(E,T=10):
    E=E.astype(dtype=np.float128)
    return 1/(1+np.exp(E/(kb*T)))


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
    
def path_create(n_points,corners):
    distance = np.zeros(shape=(len(corners)))
    for i in range(len(corners)):
        if i==0:
            distance[i]=0
        else:
            distance[i]= np.linalg.norm(corners[i-1]-corners[i])
    point_ratio = distance/np.sum(distance)
    total_points = (n_points*point_ratio).round(0).astype(int)
    total_number = np.sum(total_points).astype(int)
    corner_points = np.zeros(len(corners),dtype=int)
    temp=0
    for i in range(len(corners)):
        temp+=total_points[i]
        corner_points[i]=temp
    # print(corner_points)
    counter=0

    counter=0
    path=np.zeros(shape=(n_points+1,3))
    for i in range(1,len(corners)):
        temp = corners[i-1].copy()
        # print(i,counter,temp)
        # path.append(temp)
        path[counter]=temp
        counter+=1
        for j in range(total_points[i]-1):
            temp+=(corners[i]-corners[i-1])/total_points[i]
            # print(i,counter,temp)    
            # path.append(temp)
            path[counter]=temp
            counter+=1
        if i==len(corners)-1:
            temp+=(corners[i]-corners[i-1])/total_points[i]
            # print(i,counter,temp)
            # path.append(temp)
            path[counter]=temp
            counter+=1
    return(corner_points,path)


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

class epw:
    def __init__(self,work_dir,outfolder,nk,nq,nph,ef):
        #Initialize data array
        self.ph = np.zeros(shape=(nph,nq))
        self.g_abs = np.zeros(shape=(nph,nq,nk))
        self.g_complex = np.zeros(shape=(nph,nq,nk))
        self.g_re = np.zeros(shape=(nph,nq,nk))
        self.g_im = np.zeros(shape=(nph,nq,nk))
        self.e_k = np.zeros(shape=(nq,nk))
        self.e_kq = np.zeros(shape=(nq,nk))
        self.nk = nk
        self.nq = nq
        self.nph = nph
        self.ef=ef
        self.work_dir = work_dir
        self.outfolder=outfolder
    def load_data(self):        
        ph_df = pd.DataFrame()
        g_abs_df = pd.DataFrame()
        g_re_df = pd.DataFrame()
        g_im_df = pd.DataFrame()
        e_k_df = pd.DataFrame()
        e_kq_df = pd.DataFrame()
        for i in range(self.nph):
            ph_df[i]    = pd.read_csv(f"{self.work_dir}/{self.outfolder}/omega/omega_{i+1}.dat", delimiter=' ',header=None)
            g_abs_df[i] = pd.read_csv(f"{self.work_dir}/{self.outfolder}/gkk_abs/gkk_{i+1}.dat", delimiter=' ',header=None)
            g_re_df[i]  = pd.read_csv(f"{self.work_dir}/{self.outfolder}/gkk_re/gkk_{i+1}.dat", delimiter=' ',header=None)
            g_im_df[i]  = pd.read_csv(f"{self.work_dir}/{self.outfolder}/gkk_im/gkk_{i+1}.dat", delimiter=' ',header=None)
        e_k_df  = pd.read_csv(f"{self.work_dir}/{self.outfolder}/enk/enk_{1}.dat", delimiter=' ',header=None)
        e_kq_df = pd.read_csv(f"{self.work_dir}/{self.outfolder}/enkq/enkq_{1}.dat", delimiter=' ',header=None)
        for i in range(self.nph):
            self.ph[i] = ph_df[i].values.reshape(self.nq,self.nk).T[0]
            self.g_abs[i] = g_abs_df[i].values.reshape(self.nq,self.nk)
            self.g_re[i] = g_re_df[i].values.reshape(self.nq,self.nk)
            self.g_im[i] = g_im_df[i].values.reshape(self.nq,self.nk)
        self.e_k = e_k_df.values.reshape(self.nq,self.nk)
        self.e_kq = e_kq_df.values.reshape(self.nq,self.nk)
        self.g_complex = self.g_re+1j*self.g_im
        #remove data frames to clean memmory
        del ph_df
        del g_abs_df
        del g_re_df
        del g_im_df
        del e_k_df
        del e_kq_df

    def calculate_self_energy(self,q):
        res = np.zeros(self.nph,dtype=complex)
        for n in range(self.nph):
            temp_res = 0
            for k in range(self.nk):
                temp_res += self.calculate_eph(n,q,k)*self.calculate_suscep(n,q,k)
            res[n]=temp_res/self.nk
        return res
        
    def calculate_eph(self,n,q,k):
        eph = self.g0[n][q][k]*self.g0[n][q][k].conj()*10**-6
        return eph
        
    def calculate_suscep(self,n,q,k,delta=0.0000001):
        top = fd(self.e_k[q][k]-self.ef)-fd(self.e_kq[q][k]-self.ef)
        bottom = self.e_k[q][k]-self.e_kq[q][k]-1j*delta-self.ph[n][q]*10**-3
        return top/bottom



