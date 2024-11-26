from craftsman import CraftsManPipeline
import torch

# # load from local ckpt
# pipeline = CraftsManPipeline.from_pretrained("./ckpts/craftsman", device="cuda:0", torch_dtype=torch.float32) 

# load from huggingface model hub
pipeline = CraftsManPipeline.from_pretrained("craftsman3d/craftsman", device="cuda:0", torch_dtype=torch.float32)

image_file = "val_data/werewolf.png"
obj_file = "werewolf.glb" # output obj or glb file
textured_obj_file = "werewolf_textured.glb"
# inference
mesh = pipeline(image_file).meshes[0]
mesh.export(obj_file)

########## For texture generation, currently only support the API ##########
headers = {'Content-Type': 'application/json'}
# server_url = "114.249.238.184:34119"
server_url = "algodemo.sz.lightions.top:31025"
# server_url = "algodemo.sz.lightions.top:31024"
with open(image_file, 'rb') as f:
	image_bytes = f.read()
with open(obj_file, 'rb') as f:
	mesh_bytes = f.read()
request = {
	'png_base64_image': base64.b64encode(image_bytes).decode('utf-8'),
	'glb_base64_mesh': base64.b64encode(mesh_bytes).decode('utf-8'),
}
response = requests.post(
	url=f"http://{server_url}/generate_texture", 
	headers=headers, 
	data=json.dumps(request),
).json()
mesh_bytes = base64.b64decode(response['glb_base64_mesh'])
with open(textured_obj_file, 'wb') as f:
	f.write(mesh_bytes)
