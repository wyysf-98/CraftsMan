[中文版](README_zh.md)
<p align="center">
  <img src="asset/logo.png"  height=220>
</p>

### <div align="center">CraftsMan: High-fidelity Mesh Generation <br> with 3D Native Generation and Interactive Geometry Refiner<div> 
#####  <p align="center"> [Weiyu Li<sup>*1,2</sup>](https://wyysf-98.github.io/), Jiarui Liu<sup>*1,2</sup>, [Rui Chen<sup>1,2</sup>](https://aruichen.github.io/), [Yixun Liang<sup>2,3</sup>](https://yixunliang.github.io/), [Xuelin Chen<sup>4</sup>](https://xuelin-chen.github.io/), [Ping Tan<sup>1,2</sup>](https://ece.hkust.edu.hk/pingtan), [Xiaoxiao Long<sup>1,2</sup>](https://www.xxlong.site/)</p>
#####  <p align="center"> <sup>1</sup>HKUST, <sup>2</sup>LightIllusions, <sup>3</sup>HKUST(GZ), <sup>4</sup>Tencent AI Lab</p>
<div align="center">
  <a href="https://craftsman3d.github.io/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="http://algodemo.bj.lightions.top:24926"><img src="https://www.gradio.app/_app/immutable/assets/gradio.CHB5adID.svg" height="25"/></a> &ensp;
<!--   <a class="btn" href="https://a3b7e6fd333c27deaa.gradio.live" role="button" target="_blank"> 
    <i class="fa-solid fa-chess-knight"></i> Gradio </a> &ensp; -->
  <a href="https://arxiv.org/pdf/2405.14979"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv&color=red&logo=arxiv"></a> &ensp;
</div>

#### TL; DR: <font color="red">**CraftsMan (aka 匠心)**</font> is a two-stage text/image to 3D mesh generation model. By mimicking the modeling workflow of artist/craftsman, we propose to generate a coarse mesh (5s) with smooth geometry using 3D diffusion model and then refine it (20s) using enhanced multi-view normal maps generated by 2D normal diffusion, which is also can be in a interactive manner like Zbrush.


#### Important: the released ckpt are mainly trained on character, so it would perform better in this category and we plan to release more advanced pretrained models in the future.

## ✨ Overview

This repo contains source code (training / inference) of 3D diffusion model, pretrained weights and gradio demo code of our 3D mesh generation project, you can find more visualizations on our [project page](https://craftsman3d.github.io/) and try our [demo](https://a3b7e6fd333c27deaa.gradio.live) and [tutorial](tutorial.md). If you have high-quality 3D data or some other ideas, we very much welcome any form of cooperation.
<details><summary>Full abstract here</summary>
We present a novel generative 3D modeling system, coined CraftsMan, which can generate high-fidelity 3D geometries with highly varied shapes, regular mesh topologies, and detailed surfaces, and, notably, allows for refining the geometry in an interactive manner. Despite the significant advancements in 3D generation, existing methods still struggle with lengthy optimization processes, irregular mesh topologies, noisy surfaces, and difficulties in accommodating user edits, consequently impeding their widespread adoption and implentation in 3D modeling softwares. Our work is inspired by the craftsman, who usually roughs out the holistic figure of the work first and elaborate the surface details subsequently. Specifically, we employ a 3D native diffusion model, which operates on latent space learned from latent set-based 3D representations, to generate coarse geometries with regular mesh topology in seconds. In particular, this process takes as input a text prompt or a reference image, and leverages a powerful multi-view (MV) diffusion model to generates multiple views of the coarse geometry, which are fed into our MV-conditioned 3D diffusion model for generating the 3D geometry, significantly improving robustness and generalizability. Following that, a normal-based geometry refiner is used to significantly enhance the surface details. This refinement can be performed automatically, or interactively with user-supplied edits. Extensive experiments demonstrate that our method achieves high efficiency in producing superior quality 3D assets compared to existing methods.
</details>

<p align="center">
  <img src="asset/teaser.jpg" >
</p>


## Contents
* [Pretrained Models](##-Pretrained-models)
* [Gradio & Huggingface Demo](#Gradio-demo)
* [Inference](#Inference)
* [Training](#Train)
* [Data Prepration](#data)
* [Video](#Video)
* [Acknowledgement](#Acknowledgements)
* [Citation](#Bibtex)

## Environment Setup

<details> <summary>Hardware</summary>
We train our model on 32x A800 GPUs with a batch size of 32 per GPU for 7 days.

The mesh refinement part is performed on a GTX 3080 GPU.


</details>
<details> <summary>Setup environment</summary>

:smiley: We also provide a Dockerfile for easy installation, see [Setup using Docker](./docker/README.md).

 - Python 3.10.0
 - PyTorch 2.1.0
 - Cuda Toolkit 11.8.0
 - Ubuntu 22.04

Clone this repository.

```sh
git clone https://github.com/wyysf-98/CraftsMan.git
```

Install the required packages.

```sh
conda create -n CraftsMan python=3.10
conda activate CraftsMan
conda install -c pytorch pytorch=2.3.0 torchvision=0.18.0 cudatoolkit=11.8 && \
pip install -r docker/requirements.txt
```

</details>


# 🎥 Video

[![Watch the video](asset/video_cover.png)](https://www.youtube.com/watch?v=WhEs4tS4mGo)


# 3D Native Diffusion Model (Latent Set Diffusion Model)
We provide the training and the inference code here for future research.
The latent set diffusion model is heavily build on the same structure of [Michelangelo](https://github.com/NeuralCarver/Michelangelo),
which is based on a [perceiver](https://github.com/google-deepmind/deepmind-research/blob/master/perceiver/perceiver.py) and with 104M parameters.

## Pretrained models
Currently, We provide the [models](https://huggingface.co/wyysf/CraftsMan) with 4 view images as condition and inject camera information via ModLN to the clip feature extractor.
We will consider open source the further models according to the real situation.
```bash
## you can just get the model using wget:
wegt https://huggingface.co/wyysf/CraftsMan/blob/main/image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6/config.yaml
wegt https://huggingface.co/wyysf/CraftsMan/blob/main/image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6/model.ckpt
wegt https://huggingface.co/wyysf/CraftsMan/blob/main/image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6-aligned-vae/config.yaml
wegt https://huggingface.co/wyysf/CraftsMan/blob/main/image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6-aligned-vae/model.ckpt

## or you can git clone the repo:
git lfs install
git clone https://huggingface.co/wyysf/CraftsMan

```
If you download the models using wget, you should manually put them under the `ckpts/image-to-shape-diffusion` directory.

## Gradio demo
We provide gradio demos with different text/image-to-MV diffusion models, such as [CRM](https://github.com/thu-ml/CRM), [Wonder3D](https://github.com/xxlong0/Wonder3D/) and [LGM](https://github.com/3DTopia/LGM). You can select different models to get better results. To run a gradio demo in your local machine, simply run:

```bash
python gradio_app.py --model_path ./ckpts/image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6-aligned-vae
```

## Inference
To generate 3D meshes from images folders via command line, simply run:
```bash
python inference.py --input eval_data --device 0 --model ./ckpts/image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6-aligned-vae
```

You can modify the used mv-images model by:
```bash
python inference.py --input eval_data --mv_model 'ImageDream' --device 0  # support ['CRM', 'ImageDream', 'Wonder3D'] --model ./ckpts/image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6-aligned-vae
```

We use [rembg](https://github.com/danielgatis/rembg) to segment the foreground object by default. If the input image already has an alpha mask, please specify the no_rembg flag:
```bash
python inference.py --input 'apps/examples/1_cute_girl.webp' --device 0 --no_rembg --model ./ckpts/image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6-aligned-vae
```

If you have images from other views (left, right, bacj), you can specify images by:
```bash
python inference.py --input 'apps/examples/front.webp' --device 0 --right_view 'apps/examples/right.webp' --model ./ckpts/image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6-aligned-vae
```

For more configs, please refer to the `inference.py`.

## Train from scratch
We provide our training code to facilitate future research. And we provide a data sample in `data`.
For the occupancy part, you can download from [Objaverse-MIX](https://huggingface.co/datasets/BAAI/Objaverse-MIX/tree/main) for easy use.
For more training details and configs, please refer to the `configs` folder.

```bash
### training the shape-autoencoder
python launch.py --config ./configs/shape-autoencoder/l256-e64-ne8-nd16.yaml \
                 --train --gpu 0

### training the image-to-shape diffusion model
python launch.py --config .configs/image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6.yaml \
                 --train --gpu 0

```

# 2D Normal Enhancement Diffusion Model (TBA)

We are diligently working on the release of our mesh refinement code. Your patience is appreciated as we put the final touches on this exciting development." 🔧🚀

You can also find the video of mesh refinement part in the video.


# ❓Common questions
Q: Tips to get better results.
1. CraftsMan takes multi-view images as condition of the 3D diffusion model. By our experiments, compared with the reconstruction model like ([Wonder3D](https://github.com/xxlong0/Wonder3D/), [InstantMesh](https://github.com/TencentARC/InstantMesh/tree/main)),
our method is more robust to multi-view inconsistency. As we rely on the image-to-MV model, the facing direction of input images is very important and it always leads to good reconstruction.
2. If you have your own multi-view images, it would be a good choice to use your own images rather than the generated ones
3. Just like the 2D diffusion model, try different seeds, adjust the CFG scale or different scheduler. Good Luck.
4. We will provide a version that conditioned on the text prompt, so you can use some positive and negative prompts.


# 💪 ToDo List

- [x] Inference code
- [x] Training code
- [x] Gradio & Hugging Face demo
- [x] Model zoo, we will release more ckpt in the future
- [x] Environment setup
- [x] Data sample
- [ ] Code for mesh refine


# 🤗 Acknowledgements

- Thanks to [LightIllusion](https://www.lightillusions.com/) for providing computational resources and Jianxiong Pan for data preprocessing. If you have any idea about high-quality 3D Generation, welcome to contact us!
- Thanks to [Hugging Face](https://github.com/huggingface) for sponsoring the nicely demo!
- Thanks to [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet/tree/master) for their amazing work, the latent set representation provides an efficient way to represent 3D shape!
- Thanks to [Michelangelo](https://github.com/NeuralCarver/Michelangelo) for their great work, our model structure is heavily build on this repo!
- Thanks to [CRM](https://github.com/thu-ml/CRM), [Wonder3D](https://github.com/xxlong0/Wonder3D/) and [LGM](https://github.com/3DTopia/LGM) for their released model about multi-view images generation. If you have a more advanced version and want to contribute to the community, we are welcome to update.
- Thanks to [Objaverse](https://objaverse.allenai.org/), [Objaverse-MIX](https://huggingface.co/datasets/BAAI/Objaverse-MIX/tree/main) for their open-sourced data, which help us to do many validation experiments.
- Thanks to [ThreeStudio](https://github.com/threestudio-project/threestudio) for their great repo, we follow their fantastic and easy-to-use code structure!


# 📑License
CraftsMan is under [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.en.html), so any downstream solution and products (including cloud services) that include CraftsMan code or a trained model (both pretrained or custom trained) inside it should be open-sourced to comply with the AGPL conditions. If you have any questions about the usage of CraftsMan, please contact us first.


# 📖 BibTeX

    @misc{li2024craftsman,
    title         = {CraftsMan: High-fidelity Mesh Generation with 3D Native Generation and Interactive Geometry Refiner}, 
    author        = {Weiyu Li and Jiarui Liu and Rui Chen and Yixun Liang and Xuelin Chen and Ping Tan and Xiaoxiao Long},
    year          = {2024},
    archivePrefix = {arXiv preprint arXiv:2405.14979},
    primaryClass  = {cs.CG}
    }
