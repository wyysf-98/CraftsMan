import numpy as np
import torch
import time
import nvdiffrast.torch as dr
from util.utils import get_tri
import tempfile
from mesh import Mesh
import zipfile
def generate3d(model, rgb, ccm, device):

    color_tri = torch.from_numpy(rgb)/255
    xyz_tri = torch.from_numpy(ccm[:,:,(2,1,0)])/255
    color = color_tri.permute(2,0,1)
    xyz = xyz_tri.permute(2,0,1)


    def get_imgs(color):
        # color : [C, H, W*6]
        color_list = []
        color_list.append(color[:,:,256*5:256*(1+5)])
        for i in range(0,5):
            color_list.append(color[:,:,256*i:256*(1+i)])
        return torch.stack(color_list, dim=0)# [6, C, H, W]
    
    triplane_color = get_imgs(color).permute(0,2,3,1).unsqueeze(0).to(device)# [1, 6, H, W, C]

    color = get_imgs(color)
    xyz = get_imgs(xyz)

    color = get_tri(color, dim=0, blender= True, scale = 1).unsqueeze(0)
    xyz = get_tri(xyz, dim=0, blender= True, scale = 1, fix= True).unsqueeze(0)

    triplane = torch.cat([color,xyz],dim=1).to(device)
    # 3D visualize
    model.eval()
    glctx = dr.RasterizeCudaContext()

    if model.denoising == True:
        tnew = 20
        tnew = torch.randint(tnew, tnew+1, [triplane.shape[0]], dtype=torch.long, device=triplane.device)
        noise_new = torch.randn_like(triplane) *0.5+0.5
        triplane = model.scheduler.add_noise(triplane, noise_new, tnew)    
        start_time = time.time()
        with torch.no_grad():
            triplane_feature2 = model.unet2(triplane,tnew)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"unet takes {elapsed_time}s")
    else:
        triplane_feature2 = model.unet2(triplane)
        

    with torch.no_grad():
        data_config = {
            'resolution': [1024, 1024],
            "triview_color": triplane_color.to(device),
        }

        verts, faces = model.decode(data_config, triplane_feature2)

        data_config['verts'] = verts[0]
        data_config['faces'] = faces
        

    from kiui.mesh_utils import clean_mesh
    verts, faces = clean_mesh(data_config['verts'].squeeze().cpu().numpy().astype(np.float32), data_config['faces'].squeeze().cpu().numpy().astype(np.int32), repair = False, remesh=False, remesh_size=0.005)
    data_config['verts'] = torch.from_numpy(verts).cuda().contiguous()
    data_config['faces'] = torch.from_numpy(faces).cuda().contiguous()

    start_time = time.time()
    with torch.no_grad():
        mesh_path_obj = tempfile.NamedTemporaryFile(suffix=f"", delete=False).name
        model.export_mesh_wt_uv(glctx, data_config, mesh_path_obj, "", device, res=(1024,1024), tri_fea_2=triplane_feature2)

        mesh = Mesh.load(mesh_path_obj+".obj", bound=0.9, front_dir="+z")
        mesh_path_glb = tempfile.NamedTemporaryFile(suffix=f"", delete=False).name
        mesh.write(mesh_path_glb+".glb")

        # mesh_obj2 = trimesh.load(mesh_path_glb+".glb", file_type='glb')
        # mesh_path_obj2 = tempfile.NamedTemporaryFile(suffix=f"", delete=False).name
        # mesh_obj2.export(mesh_path_obj2+".obj")

        with zipfile.ZipFile(mesh_path_obj+'.zip', 'w') as myzip:
            myzip.write(mesh_path_obj+'.obj', mesh_path_obj.split("/")[-1]+'.obj')
            myzip.write(mesh_path_obj+'.png', mesh_path_obj.split("/")[-1]+'.png')
            myzip.write(mesh_path_obj+'.mtl', mesh_path_obj.split("/")[-1]+'.mtl')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"uv takes {elapsed_time}s")
    return mesh_path_glb+".glb", mesh_path_obj+'.zip'
