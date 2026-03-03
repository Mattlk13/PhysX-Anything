<div align="left">
<h1 align="center">PhysX-Anything: Simulation-Ready Physical 3D Assets from Single Image
</h1>
<p align="center"><a href="https://arxiv.org/abs/2511.13648"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://physx-anything.github.io/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=homepage&logoColor=white' alt='Project Page'></a>
<a href='https://huggingface.co/datasets/Caoza/PhysX-Mobility'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue'></a>
<a href='https://youtu.be/okMms-NdxMk'><img src='https://img.shields.io/youtube/views/okMms-NdxMk'></a>
<div align="center">
    <a href="https://ziangcao0312.github.io/" target="_blank">Ziang Cao</a><sup>1</sup>,
     <a href="https://hongfz16.github.io/" target="_blank">Fangzhou Hong</a><sup>1</sup>,
    <a href="https://frozenburning.github.io/" target="_blank">Zhaoxi Chen</a><sup>1</sup>,
    <a href="https://github.com/paul007pl" target="_blank">Liang Pan</a><sup>2</sup>,
    <a href="https://liuziwei7.github.io/" target="_blank">Ziwei Liu</a><sup>1</sup>
</div>
<div align="center">
    <sup>1</sup>S-Lab, Nanyang Technological University&emsp; <sup>2</sup>Shanghai AI Laboratory
</div>
<div>



<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="img/teaser.png">
</div>


## 🏆 News

- PhysX-Anything has been accepted by CVPR 2026🎉
- We release the fine-tuning code of PhysX-Anything🎉
- We release the inference code of PhysX-Anything and our new dataset PhysX-Mobility 🎉

## PhysX-Anything

### Installation

1. Clone the repo:

```
git clone --recurse-submodules https://github.com/ziangcao0312/PhysX-Anything.git
cd PhysX-Anything 
```

2. Create a new conda environment named `physx-anything` and install the dependencies:

```bash
. ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
```

**Note**: The detailed usage of `setup.sh` can be found at [TRELLIS](https://github.com/microsoft/TRELLIS)

3. Install the dependencies for Qwen2.5:

```bash
pip install transformers==4.50.0
pip install qwen-vl-utils
pip install 'accelerate>=0.26.0'
```

**Note**: We release the `requirements.txt` file. You can install all dependencies by running:

```bash
conda create -n physx-anything python=3.10
conda activate physx-anything
pip install -r requirements.txt
```

### Training

1. Download PhysX datasets from [PhysXNet](https://huggingface.co/datasets/Caoza/PhysX-3D) and [PhysX-Mobility](https://huggingface.co/datasets/Caoza/PhysX-Mobility)

2. Run the preprocessing script. 

   ```python
   cd dataset
   python 1voxel.py
   python 2encode_representation_32_finetune.py
   python 3generate_data_new_32_finetune.py
   ```

   **Note**: Here is a template for you to check the format: [template](https://github.com/ziangcao0312/PhysX-Anything/blob/main/dataset/training_data_template.json).

3. Render the conditioning images (25 images per object) based on your requirements. 

   For PhysX-Mobility, we use [dataset_toolkits/render_cond_mobility.py](https://github.com/ziangcao0312/PhysX-Anything/tree/main/dataset_toolits) to generate the conditioning images. 

   For PhysXNet, please check [PhysX-3D/dataset_toolkits/precess.sh](https://github.com/ziangcao0312/PhysX-3D/blob/main/dataset_toolkits/precess.sh)

4. Set the path in train [configuration](https://github.com/ziangcao0312/PhysX-Anything/blob/main/qwen-vl-finetune/qwenvl/data/__init__.py)

   ```python
   PHYSXNET = {
       "annotation_path": "xx", #json file path
       "data_path": "xx",  # conditioning image path
   }
   
   PHYSXMOBILITY = {
       "annotation_path": "xx", #json file path
       "data_path": "xx",  # conditioning image path
   }
   ```

5. Finetune the model

   ```
   cd qwen-vl-finetune
   sbatch scripts/sft_7b.sh
   ```

### Inference

1. Download the pre-train model from [huggingface_v2](https://huggingface.co/Caoza/PhysX-Anything).

```bash
python download.py
```

2. Run the inference code

```bash
python 1_vlm_demo.py            # vlm inference
    --demo_path ./demo          # inputted image path
    --save_part_ply True        # save the geometry of parts 
    --remove_bg False           # Set this to false for RGBA images and true otherwise.
    --ckpt ./pretrain/vlm       # ckpt path
    
python 2_decoder.py             # decoder inference

python 3_split.py               # split the mesh

python 4_simready_gen.py        # convert to URDF & XML
    --voxel_define 32           # voxel resolution
    --basepath ./test_demo      # results path
    --process 0                 # use postprocess
    --fixed_base 0              # fix the basement of object or not
    --deformable 0              # introduce deformable parts or not
```

**Note**: Although our method can generate parts with physical deformable parameters, the deformable components are not stable in MuJoCo. Therefore, we recommend setting the deformable flag to 0 to obtain more reliable simulation results.

### Evaluation

1. Render the generated URDF files

```bash
python render_urdf.py
```

2. Run the VLM-based evaluations.

```bash
python evaluation_kine.py
```

3. For all other physical attributes, please run the script.

```bash
python evaluation_phy.py
```



## PhysX-Mobility

For more details about our proposed dataset including dataset structure and annotation, please see this [PhysX-Mobility](https://huggingface.co/datasets/Caoza/PhysX-Mobility) and [PhysXNet](https://huggingface.co/datasets/Caoza/PhysX-3D).

## References

If you find PhysX-Anything and PhysX-3D useful for your work, please cite:

```
@article{physxanything,
  title={PhysX-Anything: Simulation-Ready Physical 3D Assets from Single Image},
  author={Cao, Ziang and Hong, Fangzhou and Chen, Zhaoxi and Pan, Liang and Liu, Ziwei},
  journal={arXiv preprint arXiv:2511.13648},
  year={2025}
}

@article{physx3d,
  title={PhysX-3D: Physical-Grounded 3D Asset Generation},
  author={Cao, Ziang and Chen, Zhaoxi and Pan, Liang and Liu, Ziwei},
  journal={arXiv preprint arXiv:2507.12465},
  year={2025}
}
```

### Acknowledgement

The data and code is based on [PartNet-mobility](https://sapien.ucsd.edu/browse), [Qwen](https://github.com/QwenLM/Qwen3-VL) and [TRELLIS](https://github.com/microsoft/TRELLIS). We would like to express our sincere thanks to the contributors.

## :newspaper_roll: License

Distributed under the S-Lab License. See `LICENSE` for more information.

<div align="center">
  <a href="https://info.flagcounter.com/x0BB"><img src="https://s01.flagcounter.com/map/x0BB/size_s/txt_000000/border_CCCCCC/pageviews_0/viewers_0/flags_0/" alt="Flag Counter" border="0"></a>
</div>
