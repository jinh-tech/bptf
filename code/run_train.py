from subprocess import call

dataset = "icews_aaron"

data_path = "data/"+dataset+"/" + dataset +".npz"
out_path = "output/"+dataset
mask_path = "data/"+dataset+"/train_mask"

for i in range(1,11):
	print i
	call(["python" , "code/bptf.py" , "-d="+data_path , "-o="+out_path , "-k=50" , "-n=100" , "-v" , "-m="+mask_path+str(i)+".npz"])
