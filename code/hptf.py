# %load bptf.py
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
    def __init__(self, n_modes=4, n_components=100,  max_iter=200, tol=0.0001,
                 smoothness=100, verbose=True, alpha=0.1, alpha_prime=0.1, beta_prime=1.0, debug=False):
        self.n_modes = n_modes
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.smoothness = smoothness
        self.verbose = verbose
        self.debug = debug

        self.alpha = alpha                                      # shape hyperparameter
        self.alpha_prime = alpha_prime
        self.beta_prime = beta_prime

        self.gamma_DK_M = np.empty(self.n_modes, dtype=object)  # variational shapes
        self.delta_DK_M = np.empty(self.n_modes, dtype=object)  # variational rates

        self.E_DK_M = np.empty(self.n_modes, dtype=object)      # arithmetic expectations
        self.G_DK_M = np.empty(self.n_modes, dtype=object)      # geometric expectations

        self.kappa_shp = np.empty(self.n_modes,dtype=object)
        self.kappa_rte = np.empty(self.n_modes,dtype=object)        
        # Inference cache
        self.sumE_MK = np.empty((self.n_modes, self.n_components), dtype=float)
        self.zeta = None
        self.nz_recon_I = None

    def _reconstruct_nz(self, subs_I_M):
        """Computes the reconstruction for only non-zero entries."""
        I = subs_I_M[0].size
        K = self.n_components
        nz_recon_IK = np.ones((I, K))
        for m in xrange(self.n_modes):
            nz_recon_IK *= self.G_DK_M[m][subs_I_M[m], :]
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
        self.kappa_shp[m] = np.ones(dim,dtype=np.float64) * (self.alpha_prime + K*self.alpha)
        self.kappa_rte[m] = np.ones(dim,dtype=np.float64)

    def _check_component(self, m):
        assert np.isfinite(self.E_DK_M[m]).all()
        assert np.isfinite(self.G_DK_M[m]).all()
        assert np.isfinite(self.gamma_DK_M[m]).all()
        assert np.isfinite(self.delta_DK_M[m]).all()

    def _update_gamma(self, m, data):
        if isinstance(data, skt.dtensor):
            tmp = data.astype(float)
            subs_I_M = data.nonzero()
            tmp[subs_I_M] /= self._reconstruct_nz(subs_I_M)
            uttkrp_DK = tmp.uttkrp(self.G_DK_M,m)

        elif isinstance(data, skt.sptensor):
            tmp = data.vals / self._reconstruct_nz(data.subs)
            uttkrp_DK = sp_uttkrp(tmp, data.subs, m, self.G_DK_M)

        self.gamma_DK_M[m][:, :] = self.alpha + self.G_DK_M[m] * uttkrp_DK

    def _update_delta(self, m, mask=None):
        if mask is None:
            self.sumE_MK[m, :] = 1.
            uttrkp_DK = self.sumE_MK.prod(axis=0)
        else:
            uttrkp_DK = mask.uttkrp(self.E_DK_M, m)
        if uttrkp_DK.shape == (self.n_components,):
            uttrkp_DK = uttrkp_DK.reshape((1,-1))
            uttrkp_DK = uttrkp_DK.repeat(self.mode_dims[m],axis=0)
        self.delta_DK_M[m][:, :] = (self.kappa_shp[m]/self.kappa_rte[m])[:,np.newaxis] + uttrkp_DK

    def _update_kappa(self,m):
        self.kappa_rte[m] = self.beta_prime + (self.gamma_DK_M[m]/self.delta_DK_M[m]).sum(axis=1)

    def _update_cache(self, m):
        gamma_DK = self.gamma_DK_M[m]
        delta_DK = self.delta_DK_M[m]
        self.E_DK_M[m] = gamma_DK / delta_DK
        self.sumE_MK[m, :] = self.E_DK_M[m].sum(axis=0)
        self.G_DK_M[m] = np.exp(sp.psi(gamma_DK)) / delta_DK

    def _update(self, data, mask=None, modes=None):
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
                self._update_delta(m, mask)
                self._update_cache(m)
                self._update_kappa(m)
                self._check_component(m)
            # bound = self._elbo(data, mask=mask)
            bound = 0 #self.mae_nz(data)
            delta = (curr_elbo - bound) if itn > 0 else np.nan
            e = time.time() - s
            if self.verbose:
                print 'ITERATION %d:    Time: %f   Objective: %.2f    Change: %.5f'% (itn, e, bound, delta)
            #assert ((delta >= 0.0) or (itn == 0))
            curr_elbo = bound
            # if delta < self.tol:
            #     break

    def fit(self, data, mask=None):
        assert data.ndim == self.n_modes
        data = preprocess(data)
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
        nz_recon_I = self._reconstruct_nz(subs_I_M)

        return ((np.absolute(vals_I-nz_recon_I)).sum())/vals_I.size


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
    p.add_argument('-ap', '--alpha_prime', type=float, default=0.1)
    p.add_argument('-bp', '--beta_prime', type=float, default=1.0)
    p.add_argument('-v', '--verbose', action="store_true", default=False)
    p.add_argument('--debug', action="store_true", default=False)
    args = p.parse_args()

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
    
    bptf = BPTF(n_modes=data.ndim,
                n_components=args.n_components,
                max_iter=args.max_iter,
                tol=args.tol,
                smoothness=args.smoothness,
                verbose=args.verbose,
                alpha=args.alpha,
                alpha_prime=args.alpha_prime, 
                beta_prime=args.beta_prime,
                debug=args.debug)

    bptf.fit(data, mask=mask)
    e = time.time()
    print "Training time = %d"%(e-s)

    serialize_hptf(bptf, args.out, num=None, desc='trained_model')


if __name__ == '__main__':
    main()