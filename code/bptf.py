"""
Bayesian Poisson tensor factorization with variational inference.
"""
import sys
import time
import numpy as np
import numpy.random as rn
import scipy.special as sp
import sktensor as skt
from sklearn.base import BaseEstimator, TransformerMixin

from argparse import ArgumentParser
from utils import *


class BPTF(BaseEstimator, TransformerMixin):
    def __init__(self, n_modes=4, n_components=100,  max_iter=200, tol=0.0001,
                 smoothness=100, verbose=True, alpha=0.1, debug=False,out_path=None):
        self.n_modes = n_modes
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.smoothness = smoothness
        self.verbose = verbose
        self.debug = debug

        self.alpha = alpha                                      # shape hyperparameter
        self.beta_M = np.ones(self.n_modes, dtype=float)        # rate hyperparameter (inferred)

        self.gamma_DK_M = np.empty(self.n_modes, dtype=object)  # variational shapes
        self.delta_DK_M = np.empty(self.n_modes, dtype=object)  # variational rates

        self.E_DK_M = np.empty(self.n_modes, dtype=object)      # arithmetic expectations
        self.G_DK_M = np.empty(self.n_modes, dtype=object)      # geometric expectations

        # Inference cache
        self.sumE_MK = np.empty((self.n_modes, self.n_components), dtype=float)
        self.zeta = None
        self.nz_recon_I = None
        self.out_path = out_path    #path to store the txt files related to mae vals

    def _reconstruct_nz(self, subs_I_M, G_DK_M):
        """Computes the reconstruction for only non-zero entries."""
        I = subs_I_M[0].size
        K = self.n_components
        nz_recon_IK = np.ones((I, K))
        for m in xrange(self.n_modes):
            nz_recon_IK *= G_DK_M[m][subs_I_M[m], :]
        self.nz_recon_I = nz_recon_IK.sum(axis=1)
        return self.nz_recon_I

    def _init_all_components(self, mode_dims):
        assert len(mode_dims) == self.n_modes
        self.mode_dims = mode_dims
        for m, D in enumerate(mode_dims):
            self._init_component(m, D)

    def _init_component(self, m, dim):
        assert self.mode_dims[m] == dim
        K = self.n_components
        if not self.debug:
            s = self.smoothness
            gamma_DK = s * rn.gamma(s, 1. / s, size=(dim, K))
            delta_DK = s * rn.gamma(s, 1. / s, size=(dim, K))
        else:
            gamma_DK = np.ones((dim, K))
            delta_DK = np.ones((dim, K))
        self.gamma_DK_M[m] = gamma_DK
        self.delta_DK_M[m] = delta_DK
        self.E_DK_M[m] = gamma_DK / delta_DK
        self.sumE_MK[m, :] = self.E_DK_M[m].sum(axis=0)
        self.G_DK_M[m] = np.exp(sp.psi(gamma_DK) - np.log(delta_DK))
        self.beta_M[m] = 1. / self.E_DK_M[m].mean()

    def _check_component(self, m):
        assert np.isfinite(self.E_DK_M[m]).all()
        assert np.isfinite(self.G_DK_M[m]).all()
        assert np.isfinite(self.gamma_DK_M[m]).all()
        assert np.isfinite(self.delta_DK_M[m]).all()

    def _update_gamma(self, m, data):
        if isinstance(data, skt.dtensor):
            tmp = data.astype(float)
            subs_I_M = data.nonzero()
            tmp[subs_I_M] /= self._reconstruct_nz(subs_I_M,self.G_DK_M)
            uttkrp_DK = tmp.uttkrp(self.G_DK_M, m)

        elif isinstance(data, skt.sptensor):
            tmp = data.vals / self._reconstruct_nz(data.subs,self.G_DK_M)
            uttkrp_DK = sp_uttkrp(tmp, data.subs, m, self.G_DK_M)

        self.gamma_DK_M[m][:, :] = self.alpha + self.G_DK_M[m] * uttkrp_DK

    def _update_delta(self, m):
        self.sumE_MK[m, :] = 1.
        uttrkp_DK = self.sumE_MK.prod(axis=0)
        self.delta_DK_M[m][:, :] = self.alpha * self.beta_M[m] + uttrkp_DK

    def _update_cache(self, m):
        gamma_DK = self.gamma_DK_M[m]
        delta_DK = self.delta_DK_M[m]
        self.E_DK_M[m] = gamma_DK / delta_DK
        self.sumE_MK[m, :] = self.E_DK_M[m].sum(axis=0)
        self.G_DK_M[m] = np.exp(sp.psi(gamma_DK)) / delta_DK

    def _update_beta(self, m):
        self.beta_M[m] = 1. / self.E_DK_M[m].mean()

    def _update(self, data,orig_data=None ,modes=None, mask_no=None):
        if modes is not None:
            modes = list(set(modes))
        else:
            modes = range(self.n_modes)
        assert all(m in range(self.n_modes) for m in modes)

        curr_elbo = -np.inf
        for itn in xrange(self.max_iter):
            s = time.time()
            for m in modes:
                self._update_gamma(m, data)
                self._update_delta(m)
                self._update_cache(m)
                self._update_beta(m)  # must come after cache update!
                self._check_component(m)

            bound = self.mae_nz(data)
            delta = (curr_elbo - bound) if itn > 0 else np.nan
            e = time.time() - s
            if self.verbose:
                print 'ITERATION %d:    Time: %f   Objective: %.2f    Change: %.5f'% (itn, e, bound, delta)

            curr_elbo = bound
            # if delta < self.tol:
            #     break
            if self.bool_test==True and itn%self.test_after == 0:
                self.result_vals[2,0],self.result_vals[2,1] = self._test(orig_data,self.ind_list_t_top1,self.top1,'c')
                self.result_vals[3,0],self.result_vals[3,1] = self._test(orig_data,self.ind_list_t_top2,self.top2,'c')
                self.result_vals[0,0],self.result_vals[0,1] = self._test(orig_data,self.ind_list_c_top1,self.top1,'t')
                self.result_vals[1,0],self.result_vals[1,1] = self._test(orig_data,self.ind_list_c_top2,self.top2,'t')
                np.savetxt(self.out_path+"results_"+str(mask_no)+"_"+str(itn)+".txt",self.result_vals,fmt='%.3f')
                print self.result_vals


    def fit(self, data,test_times=None,orig_data=None,mask_no=None,bool_test=False):
        assert data.ndim == self.n_modes
        data = preprocess(data)
        self.bool_test = bool_test

        if self.bool_test == True:
            self.test_after = 5
            self.test_times = test_times
            self.nonzero_test = orig_data.nonzero()
            self.top1 = 25   # portions the test data into denser and less denser
            self.top2 = 100
            self.result_vals = np.zeros((4,2))
            self.ind_list_t_top1 = [[],[],[],[]] # contains indices for non zero entries in top*top left most indices
            self.ind_list_c_top1 = [[],[],[],[]] # contains indices for non zero entries in complement of above
            self.ind_list_t_top2 = [[],[],[],[]]
            self.ind_list_c_top2 = [[],[],[],[]]
            for i in xrange(self.nonzero_test[0].size):
                if(self.nonzero_test[3][i] in self.test_times):
                    
                    if self.nonzero_test[0][i]<self.top1 and self.nonzero_test[1][i]<self.top1:
                        for m in xrange(self.n_modes):
                            self.ind_list_t_top1[m].append(self.nonzero_test[m][i])
                    else:
                        for m in xrange(self.n_modes):
                            self.ind_list_c_top1[m].append(self.nonzero_test[m][i])
                    
                    if self.nonzero_test[0][i]<self.top2 and self.nonzero_test[1][i]<self.top2:
                        for m in xrange(self.n_modes):
                            self.ind_list_t_top2[m].append(self.nonzero_test[m][i])
                    else:
                        for m in xrange(self.n_modes):
                            self.ind_list_c_top2[m].append(self.nonzero_test[m][i])

            for m in range(self.n_modes):
                self.ind_list_t_top1[m] = np.array(self.ind_list_t_top1[m],dtype=np.int32) 
                self.ind_list_c_top1[m] = np.array(self.ind_list_c_top1[m],dtype=np.int32)
                self.ind_list_t_top2[m] = np.array(self.ind_list_t_top2[m],dtype=np.int32)
                self.ind_list_c_top2[m] = np.array(self.ind_list_c_top2[m],dtype=np.int32)

        self._init_all_components(data.shape)
        self._update(data,orig_data,None,mask_no)
        return self

    def mae_nz(self,data):

        if isinstance(data, skt.dtensor):
            subs_I_M = data.nonzero()
            vals_I = data[subs_I_M]
        elif isinstance(data, skt.sptensor):
            subs_I_M = data.subs
            vals_I = data.vals
        nz_recon_I = self._reconstruct_nz(subs_I_M,self.G_DK_M)

        return ((np.absolute(vals_I-nz_recon_I)).sum())/vals_I.size

    def _test(self,orig_data,ind_list,top,portion):
        
        max_iter = 20
        gamma_DK_M = np.copy(self.gamma_DK_M)
        delta_DK_M = np.copy(self.delta_DK_M)
        G_DK_M = np.copy(self.G_DK_M)
        E_DK_M = np.copy(self.E_DK_M)
        sumE_MK = np.copy(self.sumE_MK)
        beta_M = np.copy(self.beta_M)

        for i in xrange(0,max_iter):
            tmp = orig_data[ind_list]/self._reconstruct_nz(ind_list,G_DK_M)
            uttkrp_DK = sp_uttkrp(tmp,ind_list,3,G_DK_M)
            gamma_DK_M[3][:, :] = self.alpha + G_DK_M[3] * uttkrp_DK
            
            sumE_MK[3, :] = 1.
            uttrkp_DK = sumE_MK.prod(axis=0)
            delta_DK_M[3][:, :] = self.alpha * beta_M[3] + uttrkp_DK
            
            E_DK_M[3] = gamma_DK_M[3] / delta_DK_M[3]
            sumE_MK[3, :] = E_DK_M[3].sum(axis=0)
            G_DK_M[3] = np.exp(sp.psi(gamma_DK_M[3])) / delta_DK_M[3]
            
            beta_M[3] = 1. / E_DK_M[3].mean()

        Y_pred = parafac([G_DK_M[0],G_DK_M[1],G_DK_M[2],G_DK_M[3]])
        
        if portion=='t':
        
            Y_pred = Y_pred[:top,:top,:,self.test_times]
            temp_data = orig_data[:top,:top,:,self.test_times]
            temp_data = Y_pred - temp_data
            nz_ind = orig_data[:top,:top,:,self.test_times].nonzero()
            mae = (np.absolute(temp_data).sum())/temp_data.size
            mae_nz = (np.absolute(temp_data[nz_ind]).sum())/nz_ind[0].size   
            return mae,mae_nz
        
        else:
            temp_data1 = Y_pred[:top,top:,:,self.test_times]
            temp_data2 = Y_pred[top:,:,:,self.test_times]
            temp_data1 = orig_data[:top,top:,:,self.test_times]-temp_data1
            temp_data2 = orig_data[top:,:,:,self.test_times]-temp_data2
            nz_ind1 = orig_data[:top,top:,:,self.test_times].nonzero()
            nz_ind2 = orig_data[top:,:,:,self.test_times].nonzero()
            mae = (np.absolute(temp_data1).sum()+np.absolute(temp_data2).sum())/(temp_data1.size + temp_data2.size)
            mae_nz =  (np.absolute(temp_data1[nz_ind1]).sum()+np.absolute(temp_data2[nz_ind2]).sum())/(nz_ind1[0].size + nz_ind2[0].size)
            return mae,mae_nz        

def main():
    p = ArgumentParser()
    p.add_argument('-d', '--data', type=path, required=True)
    p.add_argument('-o', '--out', type=path, required=True)
    p.add_argument('-m', '--mask', type=path, default=None)
    p.add_argument('-k', '--n_components', type=int, required=True)
    p.add_argument('-n', '--max_iter', type=int, default=200)
    p.add_argument('-t', '--tol', type=float, default=1e-4)
    p.add_argument('-s', '--smoothness', type=int, default=100)
    p.add_argument('-a', '--alpha', type=float, default=0.1)
    p.add_argument('-v', '--verbose', action="store_true", default=False)
    p.add_argument('--debug', action="store_true", default=False)
    p.add_argument('-test','--test',type=bool, default=False)
    args = p.parse_args()

    args.out.makedirs_p()
    assert args.data.exists() and args.out.exists()
    if args.data.ext == '.npz':
        data_dict = np.load(args.data)
        if 'data' in data_dict.files:
            data = data_dict['data']
        elif 'Y' in data_dict.files:
            data = data_dict['Y']
        if data.dtype == 'object':
            assert data.size == 1
            data = data[0]
    else:
        data = np.load(args.data)

    valid_types = [np.ndarray, skt.dtensor, skt.sptensor]
    assert any(isinstance(data, vt) for vt in valid_types)
    if args.test == True:

        for i in range(1,11):   #number of test masks
            print "Test Mask %d"%(i)
            mask = np.load(args.mask+"/train_mask"+str(i)+".npz")['data']
            test_times = np.load(args.mask+'/test_times'+str(i)+'.npz')['ind']
            assert any(isinstance(mask, vt) for vt in valid_types)
            assert mask.shape == data.shape
            new_data = data * mask
            out_path = args.out + "/"
            bptf = BPTF(n_modes=data.ndim,
                   n_components=args.n_components,
                    max_iter=args.max_iter,
                    tol=args.tol,
                    smoothness=args.smoothness,
                    verbose=args.verbose,
                    alpha=args.alpha,
                    debug=args.debug,
                    out_path=out_path)
            bptf.fit(new_data,test_times,data,i,True)
        

    else:
        s = time.time()
        bptf = BPTF(n_modes=data.ndim,
                   n_components=args.n_components,
                    max_iter=args.max_iter,
                    tol=args.tol,
                    smoothness=args.smoothness,
                    verbose=args.verbose,
                    alpha=args.alpha,
                    debug=args.debug)

        bptf.fit(data)
        e = time.time()
        print "Training time = %d"%(e-s)
        serialize_bptf(bptf, args.out, num=None, desc='trained_model')


if __name__ == '__main__':
    main()
