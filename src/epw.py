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



