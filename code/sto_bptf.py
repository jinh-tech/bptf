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

#from path import path
from argparse import ArgumentParser
from utils import *


class BPTF(BaseEstimator, TransformerMixin):
    def __init__(self,batch_size,n_modes=4, n_components=100,  max_iter=10000, tol=0.0001,
                 smoothness=100, verbose=True, alpha=0.1, debug=False, test_after = 2):
        
        self.test_after = test_after
        self.batch_size = batch_size
        self.n_modes = n_modes
        self.n_components = n_components
        self.max_iter = max_iter + 1
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

    def _reconstruct_nz(self, subs_I_M):
        """Computes the reconstruction"""
        I = subs_I_M[0].size
        K = self.n_components
        nz_recon_IK = np.ones((I, K))
        for m in xrange(self.n_modes):
            nz_recon_IK *= self.G_DK_M[m][subs_I_M[m], :]
        nz_recon_I = nz_recon_IK.sum(axis=1)
        return nz_recon_I,nz_recon_IK

    def _init_all_components(self, mode_dims):
        assert len(mode_dims) == self.n_modes
        self.mode_dims = mode_dims
        self.mode_dims_prod = 1
        for m, D in enumerate(mode_dims):
            self.mode_dims_prod *= self.mode_dims[m]
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

    def _delta_expec(self,subs_I_M,m):
        
        I = subs_I_M[0].size
        K = self.n_components
        nz_recon_IK = np.ones((I, K))
        for i in xrange(self.n_modes):
            if i != m:
                nz_recon_IK *= self.E_DK_M[i][subs_I_M[i], :]
        nz_recon_I = nz_recon_IK.sum(axis=0)
        return nz_recon_I

    def _update_gamma_delta(self, m, data,t):
        
        num_ele = self.mode_dims_prod/self.mode_dims[m]
        
        for i in range(0,self.mode_dims[m]):
            rand_ind = []
            for j in range(self.n_modes):
                if j == m:
                    rand_ind.append(np.ones(shape=self.batch_size,dtype=np.int32)*i)
                else:
                    rand_ind.append(np.random.randint(low=0,high=self.mode_dims[j],size=self.batch_size,dtype=np.int32))
            nz_recon_I,nz_recon_IK = self._reconstruct_nz(rand_ind)
            self.gamma_DK_M[m][i,:] = (1-t)*self.gamma_DK_M[m][i,:] + t*(self.alpha + ((((data[rand_ind]/nz_recon_I)[:,np.newaxis])*nz_recon_IK).sum(axis=0))*(num_ele/self.batch_size))
            self.delta_DK_M[m][i,:] = (1-t)*self.delta_DK_M[m][i,:] + t*(self.alpha*self.beta_M[m] + (num_ele/self.batch_size)*self._delta_expec(rand_ind,m))

    def _update_cache(self, m):
        gamma_DK = self.gamma_DK_M[m]
        delta_DK = self.delta_DK_M[m]
        self.E_DK_M[m] = gamma_DK / delta_DK
        self.sumE_MK[m, :] = self.E_DK_M[m].sum(axis=0)
        self.G_DK_M[m] = np.exp(sp.psi(gamma_DK)) / delta_DK

    def _update_beta(self, m):
        self.beta_M[m] = 1. / self.E_DK_M[m].mean()

    def _update(self, data, mask=None, modes=None):
        if modes is not None:
            modes = list(set(modes))
        else:
            modes = range(self.n_modes)
        assert all(m in range(self.n_modes) for m in modes)

        curr_elbo = -np.inf
        t = 5
        epsilon = 0.55
        for itn in xrange(self.max_iter):
            s = time.time()
            for m in modes:
                
                self._update_gamma_delta(m, data,np.power(t,-epsilon))
                self._update_cache(m)
                self._update_beta(m)  # must come after cache update!
                self._check_component(m)
            # bound = self._elbo(data, mask=mask)
            t += 1
            if itn%self.test_after == 0:
                bound = 0#self.mae_nz(data)
                delta = (curr_elbo - bound) if itn > 0 else np.nan
                e = time.time() - s
                
                if self.verbose:
                    print 'ITERATION %d:    Time: %f   Objective: %.2f    Change: %.5f'% (itn, e, bound, delta)
                curr_elbo = bound
                #assert ((delta >= 0.0) or (itn == 0))
                # curr_elbo = bound
                # if delta < self.tol:
                #     break

    def fit(self, data, mask=None):
        assert data.ndim == self.n_modes
        data = skt.dtensor(data)

        if mask is not None:
            mask = preprocess(mask)
            assert data.shape == mask.shape
            assert is_binary(mask)
            assert np.issubdtype(mask.dtype, int)

        self._init_all_components(data.shape)
        self._update(data, mask=mask)
        return self

    def mae_nz(self,data):

        if isinstance(data, skt.dtensor):
            subs_I_M = data.nonzero()
            vals_I = data[subs_I_M]
        elif isinstance(data, skt.sptensor):
            subs_I_M = data.subs
            vals_I = data.vals
        nz_recon_I,_ = self._reconstruct_nz(subs_I_M)

        return ((np.absolute(vals_I-nz_recon_I)).sum())/vals_I.size


def main():
    p = ArgumentParser()
    p.add_argument('-b', '--batch',type=int,default=10)
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

    batch_size = args.batch
    
    args.out.makedirs_p()
    s = time.time()
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

    mask = None
    if args.mask is not None:
        if args.mask.ext == '.npz':
            mask = np.load(args.mask)['data']
            if mask.dtype == 'object':
                assert mask.size == 1
                mask = mask[0]
        else:
            mask = np.load(args.mask)

        assert any(isinstance(mask, vt) for vt in valid_types)
        assert mask.shape == data.shape
        data = data * mask
        mask = None
    
    bptf = BPTF(batch_size=batch_size,
                n_modes=data.ndim,
                n_components=args.n_components,
                max_iter=args.max_iter,
                tol=args.tol,
                smoothness=args.smoothness,
                verbose=args.verbose,
                alpha=args.alpha,
                debug=args.debug)

    bptf.fit(data, mask=mask)
    e = time.time()
    print "Training time = %d"%(e-s)
    serialize_bptf(bptf, args.out, num=None, desc='trained_model')


if __name__ == '__main__':
    main()
