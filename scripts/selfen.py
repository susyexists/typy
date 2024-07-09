import matplotlib.pyplot as plt
import numpy as np
import typy
import pandas as pd
r_space = np.linspace(0.6,0,100,endpoint=False)
theta_space = np.linspace(0,2*np.pi,360,endpoint=False)
radial_mesh = np.array([[r*np.cos(theta),r*np.sin(theta)] for r in r_space for theta in theta_space])
rx,ry= radial_mesh.T
directory = "/scratch/s.sevim/0_NbSe2_Work/0.00565/"

nk = 100*100
nq = radial_mesh.shape[0]
i = 0


# epw_ph = np.zeros(shape=(nk,nq))
# epw_ph = np.loadtxt(f"{directory}/results/omega/omega_{i+1}.dat").reshape(nq,nk).T
# fig = plt.figure(figsize=(6,5))
# plt.scatter(rx,ry,c=epw_ph[0],s=20,cmap="jet")
# plt.colorbar()
# plt.axis('equal')
# plt.xlabel("qx")
# plt.ylabel("qy")
# plt.xticks([])
# plt.yticks([])
# plt.tight_layout()
# plt.savefig("soften_phonon.png")
# plt.close()
# plt.ylim(-0.5,0.5)
# plt.show()

epw_g = np.loadtxt(f"{directory}/results/gkk/gkk_{i+1}.dat").reshape(nq,nk).T
epw_gbar = epw_g.sum(axis=0)
fig = plt.figure(figsize=(6,5))
plt.scatter(rx,ry,c=epw_gbar,s=20,cmap="jet")
plt.axis('equal')
plt.xlabel("qx")
plt.ylabel("qy")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig("gbar.png")
plt.close()
# plt.ylim(-0.5,0.5)
# plt.show()

epw_k = np.zeros(shape=(nq))
epw_kq = np.zeros(shape=(nk,nq))
epw_k = np.loadtxt(f"/{directory}/results/enk/enk_{i+1}.dat").reshape(nq,nk).T
epw_kq = np.loadtxt(f"/{directory}/results/enkq/enkq_{i+1}.dat").reshape(nq,nk).T
    
selfen = np.zeros(shape=(nph,nq))
for i in range(nph):
    for j in range(nq):
        epc = (epw_g[i].T[j]*epw_g[i].T[j])*10**-6
        fsn = (typy.fd(epw_k.T[j])-typy.fd(epw_kq.T[j]))/(epw_k.T[j]-epw_kq.T[j])
        res=np.sum(epc*fsn)
        selfen[i][j]=res
        
        
fig = plt.figure(figsize=(6,5))
plt.scatter(rx,ry,c=selfen,s=20,cmap="jet")
# plt.colorbar()
plt.axis('equal')
plt.xlabel("qx")
plt.ylabel("qy")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig("selfen.png")
plt.close()
# plt.ylim(-0.5,0.5)
# plt.show()