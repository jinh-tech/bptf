from subprocess import call

dataset = "gdelt_aaron"
algo = 'hptf'
data_path = "data/"+dataset+"/" + dataset +".npz"
out_path = "output/"+dataset+'_'+algo
mask_path = "data/"+dataset+"/train_mask"

for i in range(1,11):
	print i
	call(["python" , "code/hptf.py" , "-d="+data_path , "-o="+out_path , "-k=50" , "-n=100" , "-v" , "-m="+mask_path+str(i)+".npz"])
