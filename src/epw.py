from . import epw_utils
from . import utils
import numpy as np
import pandas as pd
import copy

class epw:
    def __init__(self,path=False,work_dir=False,outfolder=False,nk=False,nq=False,nph=False,ef=False):
        #Initialize data array
        if path!=False:
            self.path= path
        else:
            self.path=work_dir+outfolder        
        gkk_path=self.path+"gkk.out"
        if nk!=False:
            self.nk=nk
        else:
            self.nk = epw_utils.get_nk(gkk_path)
        self.nq = epw_utils.get_nq(gkk_path)
        self.nph = epw_utils.get_nph(gkk_path)
        self.ef = epw_utils.get_ef(gkk_path)
        self.work_dir = work_dir
        self.outfolder = outfolder
        self.q_path = np.arange(self.nq)

    def load_data(self):        
        self.init_array() #Initialize numpy arrays for data
        data = utils.read_parallel(self.path)
        ph_df = data['omega']
        g_abs_df = data['gkk_abs']
        g_re_df = data['gkk_re']
        g_im_df = data['gkk_im']
        e_k_df = data['enk']
        e_kq_df = data['enkq']
    
        for i in range(self.nph):
            self.ph[i] = ph_df[i].values.reshape(self.nq,self.nk).T[0]
            self.g_abs[i] = g_abs_df[i].values.reshape(self.nq,self.nk)
            self.g_re[i] = g_re_df[i].values.reshape(self.nq,self.nk)
            self.g_im[i] = g_im_df[i].values.reshape(self.nq,self.nk)
        self.e_k = e_k_df[0].values.reshape(self.nq,self.nk)
        self.e_kq = e_kq_df[0].values.reshape(self.nq,self.nk)
        self.g_complex = self.g_re+1j*self.g_im
        #remove data frames to clean memmory
        del ph_df
        del g_abs_df
        del g_re_df
        del g_im_df
        del e_k_df
        del e_kq_df
        del data

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

    def init_array(self):
        nph = self.nph
        nq = self.nq
        nk = self.nk
        self.ph = np.zeros(shape=(nph,nq))
        self.ph = np.zeros(shape=(nph,nq))
        self.g_abs = np.zeros(shape=(nph,nq,nk))
        self.g_complex = np.zeros(shape=(nph,nq,nk))
        self.g_re = np.zeros(shape=(nph,nq,nk))
        self.g_im = np.zeros(shape=(nph,nq,nk))
        self.e_k = np.zeros(shape=(nq,nk))
        self.e_kq = np.zeros(shape=(nq,nk))
    def fix_model(self,ph_tolerance=0.15,g1_tolerance=30,g2_tolerance=10,offset=50):
        ph_xs =utils.ph_cross(self.ph,tolerance=ph_tolerance)
        temp_ph = utils.untangle(self.ph,ph_xs)
        temp_g_complex = utils.untangle(self.g_complex,ph_xs)
        temp_g_complex_mean = (temp_g_complex*temp_g_complex.conj()).mean(axis=2)
        sq = np.sqrt(temp_g_complex_mean.real)
        g_xs = utils.g_cross(sq,tolerance = g1_tolerance,offset=offset)
        sq1 = utils.untangle(sq,g_xs)
        g_xs2 = utils.g_cross(sq1,tolerance = g2_tolerance,offset=offset)
        sq2 = utils.untangle(sq1,g_xs2)
        self.g_sq_mean = sq2
        temp_ph1 = utils.untangle(temp_ph,g_xs)
        fixed_ph = utils.untangle(temp_ph1,g_xs2)
        self.ph = fixed_ph
        temp_g_complex  = utils.untangle(self.g_complex,ph_xs)
        temp_g_complex1 = utils.untangle(temp_g_complex,g_xs)
        fixed_g_complex = utils.untangle(temp_g_complex1,g_xs2)
        self.g_complex = fixed_g_complex

    def reduce_g(self):
        reduced_fixed_g_complex = copy.deepcopy(self.g_complex)
        for i in range(self.nph):
            for j in range(self.nq):
                reduced_fixed_g_complex[i][j]*=np.sqrt(self.ph[i][j])
        self.g0 = reduced_fixed_g_complex