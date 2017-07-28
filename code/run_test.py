dataset = "gdelt_aaron"

data_path = "data/"+dataset+"/" + dataset +".npz"
out_path = "output/"+dataset
mask_path = "data/"+dataset+"/train_mask"

score = []

for i in range(1,11):
	print i
	test()