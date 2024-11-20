import pandas as pd
from multiprocessing import Pool, Process, Manager, cpu_count
import numpy as np
import time
import os


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def read_data(path):
    try:
        df = pd.read_csv(path, delimiter=' ',header=None)
        file_name =path.split("/")[-2] 
        # print(f"{file_name} is done")
        return [file_name,df]
    except OSError as e:
        print(e)

def read_parallel(path):
    folders = [ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]
    file_path=[]
    for folder in folders:
        files = os.listdir(path+folder)
        for file in files:
            file_path.append(path+folder+"/"+file)
    start = time.time()
    pool = Pool(cpu_count())
    df_collection = pool.map(read_data, file_path)
    pool.close()
    pool.join()
    data_folder= []
    for i in df_collection:
        data_folder.append(i[0])
        # print(i[0].split("_")[0],i[0].split("_")[1].split(".")[0])
    uniques = np.unique(np.array(data_folder))
    data_dict = {}
    for i in uniques:
        data_dict[i]=[]
    for i in df_collection:
        data_dict[i[0]].append(i[1])
    print(f"Total time: {time.time()-start}")
    return data_dict



def hexagon():
    a = np.array([[[-1/np.sqrt(3), 1/np.sqrt(3)], [1, 1]],
                  [[1/np.sqrt(3), 2/np.sqrt(3)], [1, 0]],
                  [[2/np.sqrt(3), 1/np.sqrt(3)], [0, -1]],
                  [[1/np.sqrt(3), -1/np.sqrt(3)], [-1, -1]],
                  [[-1/np.sqrt(3), -2/np.sqrt(3)], [-1, 0]],
                  [[-2/np.sqrt(3), -1/np.sqrt(3)], [0, 1]],
                  ])
    return (a)


def suscep_epw(self, point, mesh, mesh_energy, mesh_fermi, epw1, epw2=[]):
    if epw2 == []:
        epw2 = np.copy(epw1)
    shifted_energy = self.parallel_solver(point+mesh)[6]
    shifted_fermi = self.fermi(shifted_energy)
    num = mesh_fermi-shifted_fermi
    den = mesh_energy-shifted_energy
    mult = -epw1*np.conj(epw2)*num/den*10**-3
    res = np.average(mult)
    return(res)


def read_poscar(filename):
    file = open(filename, 'r')
    file.readline()
    scale = float(file.readline())
    a1 = file.readline().split()
    a2 = file.readline().split()
    a3 = file.readline().split()
    a = [a1, a2, a3]
    a = np.array(a, dtype='float') * scale
    file.close()
    return a


def crystal2reciprocal(a):  # this funciton is to get reciprocal lattice from primitive lattice

    v = np.dot(a[0], np.cross(a[1], a[2]))
    b = []
    b.append(2 * np.pi * np.cross(a[1], a[2]) / v)
    b.append(2 * np.pi * np.cross(a[2], a[0]) / v)
    b.append(2 * np.pi * np.cross(a[0], a[1]) / v)
    b = np.array(b, dtype='float')
    return b

def write_data(path,data,data_label):
    # Writing to file
    with open(path, "w") as file:
        # Writing data to a file
        file.write(f"#{' '.join(data_label)} \n")
    with open(path, "a") as file:
            for i in data:
                for j in i:
                    file.write(f"{j} ")
                file.write("\n")

def path_create(n_points,corners):
    corners=np.array(corners)
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
        path[counter]=temp
        counter+=1
        for j in range(total_points[i]-1):
            temp+=(corners[i]-corners[i-1])/total_points[i]
            path[counter]=temp
            counter+=1
        if i==len(corners)-1:
            temp+=(corners[i]-corners[i-1])/total_points[i]
            try:
                path[counter]=temp
                counter+=1
            except:
                pass
        if corner_points[-1]> n_points:
          corner_points[-1]=n_points
    return(corner_points,path)

def mesh_cartesian(num_points=[6,6,6], factor=1):
    x = np.linspace(0, 1, num_points[0])
    y = np.linspace(0, 1, num_points[1])
    z = np.linspace(0, 1, num_points[2])
    three_dim = np.array([[i, j,k] for i in x for j in y for k in z])
    return (three_dim*factor)

def fermi_bands(bands, window=0.5):
    near_bands = []
    for j,i in enumerate(bands):
        for e in i:
            if -0.5 <= e and e <= 0.5:
                near_bands.append(j)
    return set(near_bands)


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


def phselfen(self, λ, λ_real, γ, γ_real, ω):
    λ = loadtxt(self.path+"lambda.dat").reshape(-1, 9).T
    λ_real = loadtxt(self.path+"lambda_re.dat").reshape(-1, 9).T
    γ = loadtxt(self.path+"gamma.dat").reshape(-1, 9).T
    γ_real = loadtxt(self.path+"gamma_re.dat").reshape(-1, 9).T
    ω = loadtxt(self.path+"omega.dat").reshape(-1, 9).T
    return λ, λ_real, γ, γ_real, ω



