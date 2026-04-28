import os
import clip
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
from trellis.utils import render_utils
import torch.nn.functional as F
from matplotlib import cm
from matplotlib.colors import ListedColormap
import cv2
import logging
import ipdb
import trimesh
import json
from trellis.representations.mesh.cube2mesh import MeshExtractResult

def load_obj_geometry_fast(path):
    V, F = [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                x, y, z = line.split()[1:4]
                V.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                idx = [int(tok.split('/')[0]) - 1 for tok in line.split()[1:]]
                F.append(idx)
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    return trimesh.Trimesh(V, F, process=False)
def psnr(a: np.ndarray, b: np.ndarray, data_range: float | None = None) -> float:

    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"shape: {a.shape} vs {b.shape}")
    if a.ndim != 2:
        raise ValueError(f"ndims={a.ndim}")


    if data_range is None:
        if np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer):
            data_range = max(np.iinfo(a.dtype).max, np.iinfo(b.dtype).max)
        else:
            data_range = 1.0  

    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)

    mse = np.mean((a - b) ** 2)
    if mse == 0.0:
        return 50
    return 10.0 * np.log10((data_range ** 2) / mse)

def mov(mesh):
    bbox_max=np.array(mesh.vertices).max(0)
    bbox_min=np.array(mesh.vertices).min(0)
    scale = 1 / max(bbox_max - bbox_min)
    offset = -(bbox_min + bbox_max) / 2
    mesh.apply_transform(trimesh.transformations.scale_matrix(scale))
    mesh.apply_translation([offset[0],offset[1],offset[2]])
    return mesh




def draw_heatmap(data,max=1,min=0.0):
    fig = plt.figure(figsize=(7, 5))  
    ax = fig.add_subplot(111)
    cax = fig.add_axes([0.9, 0.15, 0.03, 0.7])  
    jet = cm.get_cmap('jet', 256)
    jet_colors = jet(np.linspace(0, 1, 256))
    jet_colors[0] = [0, 0, 0, 1]  
    jet_black_bg = ListedColormap(jet_colors)

    # initialize heatmap
    im = ax.imshow(np.random.rand(512,512), cmap=jet_black_bg, vmin=0, vmax=1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels([str(min), str(0.5*(max-min)), str(1.0*(max))])
    ax = fig.add_axes([0.08, 0.15, 0.8, 0.7]) 
    ax.axis('off')

    im.set_data(data)
    fig.canvas.draw()  
    img_array = np.array(fig.canvas.renderer.buffer_rgba())[..., :3]
    return img_array


resultpath='test_demo' #results path
datasetpath='./PhysX_mobility'
jsonpath=os.path.join(datasetpath,'finaljson')
meshpath=os.path.join(datasetpath,'partseg')
namelist=np.load('./val_test_list.npy') #testlist

allscale=[]
allaffordance=[]
allmaterial=[]
alldescription=[]
clipmodel, preprocess = clip.load("ViT-L/14", jit=False)
clipmodel=clipmodel.eval().cuda()
for name in namelist:
    with open(os.path.join(jsonpath,name+'.json'),'r') as fp:
        jsongtdata=json.load(fp)

    with open(os.path.join(resultpath,name,'basic_info.json'),'r') as fp:
        jsonevaldata=json.load(fp)


    index=0

    str_list=jsongtdata['dimension'].split(' ')[0].split('*')
    sorted_list = sorted(str_list, key=float, reverse=True)
    scaling_gt=float(sorted_list[0])

    str_list=jsonevaldata['dimension'].split(' ')[0].split('*')
    sorted_list = sorted(str_list, key=float, reverse=True)
    scaling_eval=float(sorted_list[0])

    dim_error=np.sqrt((scaling_gt-scaling_eval)**2)
    allscale.append(dim_error)



    allrenobj_gt=trimesh.Trimesh([])
    allsourcedata_gt=np.zeros((0,3)) 

    
    description_ind=np.random.randint(len(jsongtdata['parts']))
    text_description=jsongtdata['parts'][description_ind]['Basic_description']
    tokens=clip.tokenize(text_description).cuda()
    info_emb_gt=clipmodel.encode_text(tokens).float()

    for part in range(len(jsongtdata['parts'])):

        objlist=jsongtdata['parts'][part]['obj']

        eachobj_gt=trimesh.Trimesh([])

        for objfile in objlist:
            eachpart1=load_obj_geometry_fast(os.path.join(meshpath,name,'objs',str(objfile)+'.obj'))
            eachobj_gt = trimesh.util.concatenate([eachpart1,eachobj_gt])


        allrenobj_gt = trimesh.util.concatenate([eachobj_gt,allrenobj_gt])

        sourcedata=np.zeros((len(eachobj_gt.vertices),3))
        sourcedata[:,0]=jsongtdata['parts'][part]['priority_rank']
        sourcedata[:,1]=float(jsongtdata['parts'][part]['density'].split(' ')[0])
        if part==description_ind:
            sourcedata[:,2]=1
        else:
            sourcedata[:,2]=0
        allsourcedata_gt=np.concatenate([sourcedata,allsourcedata_gt])

    allrenobj_eval=trimesh.Trimesh([])
    allsourcedata_eval=np.zeros((0,3)) 

    description_ind_evallist=[]

    for part in range(len(jsonevaldata['parts'])):
        tokens=clip.tokenize(jsonevaldata['parts'][part]['Basic_description']).cuda()
        info_emb_eval=clipmodel.encode_text(tokens).float()
        score=F.cosine_similarity(info_emb_eval, info_emb_gt, dim=1)
        description_ind_evallist.append(score)

    description_ind_eval=int(torch.cat(description_ind_evallist).cpu().argmax()) 

    for part in range(len(jsonevaldata['parts'])):

        eachpart1=load_obj_geometry_fast(os.path.join(resultpath,name,'objs',str(part),str(part)+'.obj'))
        allrenobj_eval = trimesh.util.concatenate([eachpart1,allrenobj_eval])

        sourcedata=np.zeros((len(eachpart1.vertices),3))
        sourcedata[:,0]=jsonevaldata['parts'][part]['priority_rank']
        sourcedata[:,1]=float(jsonevaldata['parts'][part]['density'].split(' ')[0])
        if part==description_ind_eval:
            sourcedata[:,2]=1
        else:
            sourcedata[:,2]=0
        allsourcedata_eval=np.concatenate([sourcedata,allsourcedata_eval])


    rotation_matrix = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
    allrenobj_gt.apply_transform(rotation_matrix)

    allrenobj_gt=mov(allrenobj_gt)
    allrenobj_eval=mov(allrenobj_eval)
    

    gtmesh=MeshExtractResult(
        torch.Tensor(allrenobj_gt.vertices).cuda(),
        torch.Tensor(allrenobj_gt.faces).cuda(),
        vertex_attrs=None,
        res=64,
        render_vis=torch.Tensor(allsourcedata_gt).cuda()
    )
    evalmesh=MeshExtractResult(
        torch.Tensor(allrenobj_eval.vertices).cuda(),
        torch.Tensor(allrenobj_eval.faces).cuda(),
        vertex_attrs=None,
        res=64,
        render_vis=torch.Tensor(allsourcedata_eval).cuda()
    )

    video_gt = render_utils.render_video_gt(gtmesh,num_frames=30)
    video_eval = render_utils.render_video_gt(evalmesh,num_frames=30)
    


    for i in range(len(video_gt['rendervis'])):

        vis=video_gt['rendervis'][i]*video_gt['mask'][i]
        img_0=(vis[0].detach().cpu().numpy())
        img_1=(vis[1].detach().cpu().numpy())
        img_2=(vis[2].detach().cpu().numpy())

        vis_eval=video_eval['rendervis'][i]*video_eval['mask'][i]
        img_0_eval=(vis_eval[0].detach().cpu().numpy())
        img_1_eval=(vis_eval[1].detach().cpu().numpy())
        img_2_eval=(vis_eval[2].detach().cpu().numpy())
        

        allaffordance.append(psnr(img_0/img_0.max(),img_0_eval/img_0_eval.max()))
        allmaterial.append(psnr(img_1/img_1.max(),img_1_eval/img_1_eval.max()))
        alldescription.append(psnr(img_2,img_2_eval))



print('scale: ',np.array(allscale).mean())
print('affordance: ',np.array(allaffordance).mean())
print('material: ',np.array(allmaterial).mean())
print('description: ',np.array(alldescription).mean())