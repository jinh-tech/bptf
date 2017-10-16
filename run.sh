#!/bin/bash
# number	algo	dataset		rank	test 	iter 	

# 1. 		bptf gdelt_aaron 	k=10 	True 	101
# 2.		bptf icews_aaron 	k=10 	True 	101


if [ $1 -eq 1 ]
	then
	echo python code/bptf.py -d=data/gdelt_aaron/gdelt_aaron.npz -o=temp_results/ -m=data/gdelt_aaron -k=10 -v --test=True -n=101
	python code/bptf.py -d=data/gdelt_aaron/gdelt_aaron.npz -o=temp_results/ -m=data/gdelt_aaron -k=10 -v --test=True -n=101	

elif [ $1 -eq 2 ]
	then
	echo python code/bptf.py -d=data/icews_aaron/icews_aaron.npz -o=temp_results/ -m=data/icews_aaron -k=10 -v --test=True -n=101
	python code/bptf.py -d=data/icews_aaron/icews_aaron.npz -o=temp_results/ -m=data/icews_aaron -k=10 -v --test=True -n=101
fi