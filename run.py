import os
from tqdm import tqdm
from inference import main as image23d
import time

DATA_FOLDER = "eval_data"
OUTPUT_FOLDER = "eval_outputs/furniture"
CKPT_FOLDER = "ckpts/image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6-aligned-vae"

# filepath = "apps/mydata/cat.png"
# image23d(filepath, OUTPUT_FOLDER, CKPT_FOLDER, mv_model="Era3D")
filepaths = [os.path.join(DATA_FOLDER, filename) for filename in os.listdir(DATA_FOLDER)]
with tqdm(total=len(filepaths)) as pbar:
	for filepath in filepaths:
		pbar.set_description(filepath)
		start = time.time()
		image23d(filepath, OUTPUT_FOLDER, CKPT_FOLDER, mv_model="Era3D")
		pbar.update(1)
		pbar.set_description(f"{filepath}: {time.time()-start:.2f}s")
