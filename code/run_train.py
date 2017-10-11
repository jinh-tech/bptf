from subprocess import call
import os
from multiprocessing import Process, Lock
import multiprocessing as mp

def call_hptf(l,algo,data_path,out_path,mask_path,s,alpha,beta):
	l.acquire()
	try:
		call(["python" , "code/"+algo+".py" , "-d="+data_path , "-o="+out_path , "-k=50" , "-n=200" , "-v" , "-m="+mask_path ,"-test=True", "-ap="+str(alpha),"-s="+str(s),"-bp="+str(beta)])
	finally:
		l.release()

def call_sto_hptf(l,algo,data_path,out_path,mask_path,s,alpha,beta):
	l.acquire()
	try:
		call(["python" , "code/"+algo+".py" , "-d="+data_path , "-o="+out_path , "-k=50" , "-n=200" , "-v" , "-m="+mask_path ,"-test=True","-b=1000" , "-ap="+str(alpha),"-s="+str(s),"-bp="+str(beta)])
	finally:
		l.release()

def call_bptf(l,algo,data_path,out_path,mask_path,s,alpha):
	l.acquire()
	try:
		call(["python" , "code/"+algo+".py" , "-d="+data_path , "-o="+out_path , "-k=50" , "-n=200" , "-v" , "-m="+mask_path ,"-test=True", "-ap="+str(alpha),"-s"+str(s)])
	finally:
		l.release()

def call_sto_bptf(l,algo,data_path,out_path,mask_path,s,alpha):
	l.acquire()
	try:
		call(["python" , "code/"+algo+".py" , "-d="+data_path , "-o="+out_path , "-k=50" , "-n=200" , "-v" , "-m="+mask_path ,"-test=True","-b=1000" , "-ap="+str(alpha),"-s"+str(s)])
	finally:
		l.release()

datasets = ["gdelt_aaron","icews_aaron"]
algos = ['sto_hptf','sto_bptf','bptf','hptf']
init_vals = [1,5,10,20,50,100,150,200]	#smoothness parameter
alpha_vals = [0.01,0.05,0.1,0.5,1,2]  #values for alpha, alpha'
beta_vals = [0.05,0.1,0.5,1,2,5] #values for beta
num_process = 7
locks = [Lock() for i in range(0,num_process)]
counter = 0

for dataset in datasets:
	for algo in algos:
		data_path = "data/"+dataset+"/" + dataset +".npz"
		out_path = "output/"+dataset+'_'+algo
		mask_path = "data/"+dataset
		if not os.path.exists(out_path):
			os.makedirs(out_path)
		print "Number of processes = " + str(len(mp.active_children()))
		for s in init_vals:
			for alpha in alpha_vals:
				if algo == 'bptf':
					Process(target=call_bptf,args=(locks[counter%num_process],algo,data_path,out_path,mask_path,i,s,alpha)).start()
					counter += 1
				elif algo == 'sto_bptf':
					Process(target=call_sto_bptf,args=(locks[counter%num_process],algo,data_path,out_path,mask_path,i,s,alpha)).start()
					counter += 1
				for beta in beta_vals:
					if algo == 'hptf':
						Process(target=call_hptf,args=(locks[counter%num_process],algo,data_path,out_path,mask_path,i,s,alpha,beta)).start()
						counter += 1
					elif algo == 'sto_hptf':
						Process(target=call_sto_hptf,args=(locks[counter%num_process],algo,data_path,out_path,mask_path,i,s,alpha,beta)).start()
						counter += 1
