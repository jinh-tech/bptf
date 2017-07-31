from utils import *
import numpy as np
import sktensor as skt
import gc
import scipy.special as sp
import time


dataset = "icews_aaron"
algo = "bptf"
data_path = "data/"+dataset+"/" + dataset +".npz"
out_path = "output/"+dataset + '_' + algo
mask_path = "data/"+dataset +"/test_times"
result_path = "results/" + dataset + '_' + algo

mae_nz = []
mae = []

class tf:

	def __init__(self,data,result,alpha,top,modes,compliment,mask_ind):
		
		self.max_iter = 20
		self.n_modes = modes
		self.mask_ind = mask_ind
		self.alpha = alpha
		self.top = top
		self.get_params(result)
		self.n_components = self.E_DK_M[0].shape[1]
		self.sumE_MK = np.empty((self.n_modes, self.n_components), dtype=float)
		self.compliment = compliment
		temp_ind = data.nonzero()
		ind_list = [[],[],[],[]]
		if self.compliment==False:
			for i in xrange(temp_ind[0].size):
				if(temp_ind[0][i]>=top and temp_ind[1][i]>=top and temp_ind[3][i] in mask_ind):
					for m in xrange(self.n_modes):
						ind_list[m].append(temp_ind[m][i])
					
		else:
			for i in xrange(temp_ind[0].size):
				if(temp_ind[0][i]<top and temp_ind[1][i]<top and temp_ind[3][i] in mask_ind):
					for m in xrange(self.n_modes):
						ind_list[m].append(temp_ind[m][i])

		
		self.data = data.astype(float)
		for m in range(self.n_modes):
			self.sumE_MK[m, :] = self.E_DK_M[m].sum(axis=0)
			ind_list[m] = np.array(ind_list[m])
		self.ind = tuple(ind_list)

	def get_params(self,results):
		
		self.gamma_DK_M = results['gamma_DK_M']
		self.delta_DK_M = results['delta_DK_M']
		self.E_DK_M = results['E_DK_M']
		self.G_DK_M = results['G_DK_M']
		self.beta_M = results['beta_M']

	def _update_gamma(self, m):
		
		tmp = self.data[self.ind]/self._reconstruct_nz()
		uttkrp_DK = sp_uttkrp(tmp,self.ind,m,self.G_DK_M)
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

	def _reconstruct_nz(self):

		I = self.ind[0].size
		K = self.n_components
		nz_recon_IK = np.ones((I, K))
		for m in xrange(self.n_modes):
			nz_recon_IK *= self.G_DK_M[m][self.ind[m], :]
		self.nz_recon_I = nz_recon_IK.sum(axis=1)
		return self.nz_recon_I

	def update_time(self):
		
		for itn in xrange(self.max_iter):
			s = time.time()
			self._update_gamma(3)
			self._update_delta(3)
			self._update_cache(3)
			self._update_beta(3)
			self._check_component(3)
			print "Iteration %d\t Time %f"%(itn,time.time()-s)

	def _check_component(self, m):
	
		assert np.isfinite(self.E_DK_M[m]).all()
		assert np.isfinite(self.G_DK_M[m]).all()
		assert np.isfinite(self.gamma_DK_M[m]).all()
		assert np.isfinite(self.delta_DK_M[m]).all()

	def check(self):

		Y_pred = parafac([self.G_DK_M[0],self.G_DK_M[1],self.G_DK_M[2],self.G_DK_M[3]])
		if self.compliment == False:
			Y_pred = Y_pred[:self.top,:self.top,:,self.mask_ind]
			temp_data = self.data[:self.top,:self.top,:,self.mask_ind]
			temp_data = Y_pred - temp_data
			nz_ind = self.data[:self.top,:self.top,:,self.mask_ind].nonzero()
		else:
			Y_pred = Y_pred[self.top:,self.top:,:,self.mask_ind]
			temp_data = self.data[self.top:,self.top:,:,self.mask_ind]
			temp_data = Y_pred - temp_data
			nz_ind = self.data[self.top:,self.top:,:,self.mask_ind].nonzero()
			
		# mae = 0
		# mae_nz = 0
		mae = ((np.absolute(temp_data)).sum())/temp_data.size
		mae_nz = (np.absolute(temp_data[nz_ind]).sum())/nz_ind[0].size
		return mae,mae_nz


for i in range(1,11):
	print i
	result = np.load(out_path+"/"+str(i)+"_trained_model.npz")
	data = np.load(data_path)['Y']
	mask_ind = np.load(mask_path+str(i)+".npz")['ind']
	alpha = 0.1
	top = 25
	modes = len(data.shape)
	compliment = False
	model = tf(data,result,alpha,top,modes,compliment,mask_ind)
	model.update_time()
	m,nz = model.check()
	mae.append(m)
	mae_nz.append(nz)

np.savetxt(result_path +'/'+ 'mae'+str(top)+str(compliment)[0]+'.txt',mae,fmt='%.3f')
np.savetxt(result_path +'/'+ 'mae_nz'+str(top)+str(compliment)[0]+'.txt',mae_nz,fmt='%.3f')