from subprocess import call
import os

datasets = ["gdelt_aaron","icews_aaron"]
algos = ['sto_hptf','sto_bptf']

for dataset in datasets:
	for algo in algos:
		data_path = "data/"+dataset+"/" + dataset +".npz"
		out_path = "output/"+dataset+'_'+algo
		mask_path = "data/"+dataset+"/train_mask"
		if not os.path.exists(out_path):
			os.makedirs(out_path)

		for i in range(1,11):
			print algo,dataset,i
			call(["python" , "code/"+algo+".py" , "-d="+data_path , "-o="+out_path , "-k=50" , "-n=200" , "-v" , "-m="+mask_path+str(i)+".npz" , "-b=1000"])
