from utils import *
import numpy as np
import sktensor as skt
import gc
import scipy.special as sp
import time
import os

algo = "sto_hptf"
datasets = ["gdelt_aaron","icews_aaron"]
top_val = [25,100]
compliment_val = [True,False]

class tf:

	def __init__(self,data,result,alpha,top,modes,compliment,mask_ind):
		
		self.batch_size = 1000
		self.max_iter = 100
		self.n_modes = modes
		self.mask_ind = mask_ind
		self.alpha = alpha
		self.top = top
		self.get_params(result)
		self.n_components = self.E_DK_M[0].shape[1]
		self.sumE_MK = np.empty((self.n_modes, self.n_components), dtype=float)
		self.compliment = compliment
		self.data = data.astype(float)
		self.mode_dims = np.empty((self.n_modes),dtype=int)
		self.mode_dims_prod = 1
		for m in range(self.n_modes):
			self.sumE_MK[m, :] = self.E_DK_M[m].sum(axis=0)
			self.mode_dims[m] = self.G_DK_M[m].shape[0]
			self.mode_dims_prod *= self.mode_dims[m]

		if self.compliment == False:
			self.low = top
			self.high = self.mode_dims[0]
		else:
			self.low = 0
			self.high = top


	def get_params(self,results):
		
		self.gamma_DK_M = results['gamma_DK_M']
		self.delta_DK_M = results['delta_DK_M']
		self.E_DK_M = results['E_DK_M']
		self.G_DK_M = results['G_DK_M']
		# self.beta_M = results['beta_M']	#for bptf
		self.kappa_shp = results['kappa_shp']   # for hptf
		self.kappa_rte = results['kappa_rte']	# for hptf

	def _delta_expec(self,subs_I_M,m):

		I = subs_I_M[0].size
		K = self.n_components
		nz_recon_IK = np.ones((I, K))
		for i in xrange(self.n_modes):
			if i != m:
				nz_recon_IK *= self.E_DK_M[i][subs_I_M[i], :]
		nz_recon_I = nz_recon_IK.sum(axis=0)
		return nz_recon_I

	def _update_gamma_delta(self, m, t):
	
		num_ele = self.mode_dims_prod/self.mode_dims[m]
		for i in self.mask_ind:
			rand_ind = []
			for j in range(self.n_modes):
				if j == m:
					rand_ind.append(np.ones(shape=self.batch_size,dtype=np.int32)*i)
				elif j!=2:
					rand_ind.append(np.random.randint(low=self.low,high=self.high,size=self.batch_size,dtype=np.int32))
				else:
					rand_ind.append(np.random.randint(low=0,high=self.mode_dims[j],size=self.batch_size,dtype=np.int32))
			nz_recon_I,nz_recon_IK = self._reconstruct_nz(rand_ind)
			self.gamma_DK_M[m][i,:] = (1-t)*self.gamma_DK_M[m][i,:] + t*(self.alpha + ((((data[rand_ind]/nz_recon_I)[:,np.newaxis])*nz_recon_IK).sum(axis=0))*(num_ele/self.batch_size))
			self.delta_DK_M[m][i,:] = (1-t)*self.delta_DK_M[m][i,:] + t*(self.alpha*(self.kappa_shp[m][i]/self.kappa_rte[m][i]) + (num_ele/self.batch_size)*self._delta_expec(rand_ind,m)) #for hptf
			# self.delta_DK_M[m][i,:] = (1-t)*self.delta_DK_M[m][i,:] + t*(self.alpha*self.beta_M[m] + (num_ele/self.batch_size)*self._delta_expec(rand_ind,m))	#for bptf


	# def _update_gamma(self, m):
		
	# 	tmp = self.data[self.ind]/self._reconstruct_nz()
	# 	uttkrp_DK = sp_uttkrp(tmp,self.ind,m,self.G_DK_M)
	# 	self.gamma_DK_M[m][:, :] = self.alpha + self.G_DK_M[m] * uttkrp_DK

	# def _update_delta(self, m):
		
	# 	self.sumE_MK[m, :] = 1.
	# 	uttrkp_DK = self.sumE_MK.prod(axis=0)
	# 	self.delta_DK_M[m][:, :] = self.alpha * self.beta_M[m] + uttrkp_DK # for bptf
	# 	# if uttrkp_DK.shape == (self.n_components,):		#for hptf
	# 		# uttrkp_DK = uttrkp_DK.reshape((1,-1))	#for hptf
	# 		# uttrkp_DK = uttrkp_DK.repeat(self.mode_dims[m],axis=0)	#for hptf
	# 	# self.delta_DK_M[m][:, :] = self.alpha*(self.kappa_shp[m]/self.kappa_rte[m])[:,np.newaxis] + uttrkp_DK 	#for hptf


	def _update_cache(self, m):
		
		gamma_DK = self.gamma_DK_M[m]
		delta_DK = self.delta_DK_M[m]
		self.E_DK_M[m] = gamma_DK / delta_DK
		self.sumE_MK[m, :] = self.E_DK_M[m].sum(axis=0)
		self.G_DK_M[m] = np.exp(sp.psi(gamma_DK)) / delta_DK

	# def _update_beta(self, m):		#for bptf
		# self.beta_M[m] = 1. / self.E_DK_M[m].mean()		#for bptf

	def _update_kappa(self,m,t):		#for hptf
		self.kappa_rte[m] = (1-t)*self.kappa_rte[m] + t*(self.alpha + (self.gamma_DK_M[m]/self.delta_DK_M[m]).sum(axis=1))		#for hptf

	def _reconstruct_nz(self, subs_I_M):

		I = subs_I_M[0].size
		K = self.n_components
		nz_recon_IK = np.ones((I, K))
		for m in xrange(self.n_modes):
			nz_recon_IK *= self.G_DK_M[m][subs_I_M[m], :]
		nz_recon_I = nz_recon_IK.sum(axis=1)
		return nz_recon_I,nz_recon_IK

	def update_time(self):

		t = 5
		epsilon = 0.7
		for itn in xrange(self.max_iter):
			s = time.time()
			self._update_gamma_delta(3,np.power(t,-epsilon))
			self._update_cache(3)
			# self._update_beta(3)	#for bptf
			self._update_kappa(3,np.power(t,-epsilon)) #for hptf
			self._check_component(3)
			print "Iteration %d\t Time %f"%(itn,time.time()-s)
			t += 1

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
		mae = (np.absolute(temp_data).sum())/temp_data.size
		mae_nz = (np.absolute(temp_data[nz_ind]).sum())/nz_ind[0].size
		return mae,mae_nz

for dataset in datasets:

	data_path = "data/"+dataset+"/" + dataset +".npz"
	out_path = "output/"+dataset + '_' + algo
	mask_path = "data/"+dataset +"/test_times"
	result_path = "results/" + dataset + '_' + algo
	data = np.load(data_path)['Y']
	if not os.path.exists(result_path):
		os.makedirs(result_path)
	for top in top_val:
		for compliment in compliment_val:
			mae_nz = []
			mae = []
			for i in range(1,11):
				print dataset,top,compliment,i
				result = np.load(out_path+"/"+str(i)+"_trained_model.npz")
				mask_ind = np.load(mask_path+str(i)+".npz")['ind']
				alpha = 0.1
				modes = len(data.shape)
				model = tf(data,result,alpha,top,modes,compliment,mask_ind)
				model.update_time()
				m,nz = model.check()
				mae.append(m)
				mae_nz.append(nz)
			np.savetxt(result_path +'/'+ 'mae'+str(top)+str(compliment)[0]+'.txt',mae,fmt='%.3f')
			np.savetxt(result_path +'/'+ 'mae_nz'+str(top)+str(compliment)[0]+'.txt',mae_nz,fmt='%.3f')