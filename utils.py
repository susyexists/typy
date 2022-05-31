
def phselfen(self, λ, λ_real, γ, γ_real, ω):
    λ = loadtxt(self.path+"lambda.dat").reshape(-1, 9).T
    λ_real = loadtxt(self.path+"lambda_re.dat").reshape(-1, 9).T
    γ = loadtxt(self.path+"gamma.dat").reshape(-1, 9).T
    γ_real = loadtxt(self.path+"gamma_re.dat").reshape(-1, 9).T
    ω = loadtxt(self.path+"omega.dat").reshape(-1, 9).T
    return λ, λ_real, γ, γ_real, ω


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
