#!/bin/bash

DATA_FOLDER=apps/mydata
for filepath in $DATA_FOLDER/*; do
	filename=${filepath##*/}
	filename=${filename%.*}
	if [ -f "eval_outputs/${filename}.obj" ]; then
		continue
	fi
	echo $filepath;
	python inference.py --input $filepath --device 0 --model ./ckpts/image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6-aligned-vae
done