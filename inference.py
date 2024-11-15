from craftsman import CraftsManPipeline
import torch

# load from local ckpt
pipeline = CraftsManPipeline.from_pretrained("./ckpts/craftsman-v1-5", device="cuda:0", torch_dtype=torch.float32) 

# # load from huggingface model hub
# pipeline = CraftsManPipeline.from_pretrained("craftsman3d/craftsman-v1-5", device="cuda:0", torch_dtype=torch.float32)

# inference
mesh = pipeline("https://pub-f9073a756ec645d692ce3d171c2e1232.r2.dev/data/werewolf.png").meshes[0]
mesh.export("werewolf.obj")
