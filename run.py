import os
from tqdm import tqdm
from inference import main as image23d

DATA_FOLDER = "eval_data"
OUTPUT_FOLDER = "eval_outputs/furniture"
CKPT_FOLDER = "ckpts/image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6-aligned-vae"

filepaths = [os.path.join(DATA_FOLDER, filename) for filename in os.listdir(DATA_FOLDER)]
with tqdm(total=len(filepaths)) as pbar:
	for filepath in filepaths:
		pbar.set_description(filepath)
		image23d(filepath, OUTPUT_FOLDER, CKPT_FOLDER)
		pbar.update(1)
