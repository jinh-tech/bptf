
import numpy as np
from scipy.misc import logsumexp
from scipy.special import lambertw
import sktensor as skt

class poisson_response:
    
    def __init__(self,mode_dims,components,n_trunc):
        self.lam = 0.0
        self.dim = mode_dims
        self.n_trunc = n_trunc
        self.comp = components
        self.modes = len(mode_dims)

    def set_param(self,data):
        if isinstance(data, skt.dtensor):
            self.lam = data[data.nonzero()].mean()
        elif isinstance(data, skt.sptensor):
            self.lam = data.vals.mean()
        
        
    def update_expec(self,data,avgs):
        if isinstance(data, skt.dtensor):
            ind = data.nonzero()
            non_zero_ent = data[ind]
        elif isinstance(data, skt.sptensor):
            ind = data.subs
            non_zero_ent = data.vals
        
        size_ind = ind[0].size
        log_a = np.ones((size_ind,self.comp))
        
        for m in xrange(self.modes):
            log_a *= avgs[m][ind[m], :]
        log_a = np.log(log_a.sum(axis=1))
        q = np.empty((size_ind,self.n_trunc),dtype=np.float64)
        
        for i in range(1,self.n_trunc+1):
            q[:,i-1] = (-i*self.lam) + (non_zero_ent-i)*np.log(i) + i*log_a + i - 1
        norm = logsumexp(q,axis=1)
        q = np.exp(q-norm[:,np.newaxis])
        self.expec = np.zeros((size_ind,))
        for i in range(1,self.n_trunc+1):
            self.expec += i*q[:,i-1]
        
        self.expec = skt.sptensor(ind,self.expec,shape=self.dim,dtype=np.float64)
    
    # def expectation_mat(self,x,a,n_trunc):
    #     log_a = np.log(a)
        
    #     for i in range(1,n_trunc+1):
    #         self.q[:,:,i-1] = (-i*self.lam) + (x-i)*np.log(i) + i*log_a + i - 1
        
    #     norm = logsumexp(self.q,axis=2)
    #     self.q = np.exp(self.q-norm[:,:,np.newaxis])
    #     self.en = np.zeros_like(self.en)
        
    #     for i in range(1,n_trunc+1):
    #         self.en += i*self.q[:,:,i-1]
    #     self.en = np.where(x==0,0.,self.en)    
            
    #     return self.en
    
    # def sample(self,count):
    #     self.sampled = self.lam*count