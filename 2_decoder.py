import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import ipdb
import numpy as np
import torch
import trimesh
# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("./pretrain/decoder")
pipeline.cuda()

basepath='./demo'
filepath='./test_demo'
namelist=os.listdir(basepath)


for name in namelist:
    image = Image.open(os.path.join(basepath,name))
    qwenpath=os.path.join(filepath,name[:-4])

    if os.path.exists(os.path.join(qwenpath,'allind.npy')):
        newcoords=np.load(os.path.join(qwenpath,'allind.npy'))
        
        size=32
        resolution=64

        newcoords=newcoords+32-(size)//2
        
        ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
        ss[:, newcoords[:, 0], newcoords[:, 1], newcoords[:, 2]] = 1
        ss=ss.cuda().float().unsqueeze(0)



        outputs = pipeline.run_control(ss,image,seed=1,)

        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=0.5,          # Ratio of triangles to remove in the simplification process
            texture_size=1024,      # Size of the texture used for the GLB
        )


        

        
        glb.export(os.path.join(qwenpath,'sample.glb'))




