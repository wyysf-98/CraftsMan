# MVDream-diffusers

A **unified** diffusers implementation of [MVDream](https://github.com/bytedance/MVDream) and [ImageDream](https://github.com/bytedance/ImageDream).

We provide converted `fp16` weights on huggingface:
* [MVDream](https://huggingface.co/ashawkey/mvdream-sd2.1-diffusers)
* [ImageDream](https://huggingface.co/ashawkey/imagedream-ipmv-diffusers)


### Install
```bash
# dependency
pip install -r requirements.txt

# xformers is required! please refer to https://github.com/facebookresearch/xformers
pip install ninja
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```

### Usage

```bash
python run_mvdream.py "a cute owl"
python run_imagedream.py data/anya_rgba.png
```

### Convert weights

MVDream:
```bash
# download original ckpt (we only support the SD 2.1 version)
mkdir models
cd models
wget https://huggingface.co/MVDream/MVDream/resolve/main/sd-v2.1-base-4view.pt
wget https://raw.githubusercontent.com/bytedance/MVDream/main/mvdream/configs/sd-v2-base.yaml
cd ..

# convert
python convert_mvdream_to_diffusers.py --checkpoint_path models/sd-v2.1-base-4view.pt --dump_path ./weights_mvdream --original_config_file models/sd-v2-base.yaml --half --to_safetensors --test
```

ImageDream:
```bash
# download original ckpt (we only support the pixel-controller version)
cd models
wget https://huggingface.co/Peng-Wang/ImageDream/resolve/main/sd-v2.1-base-4view-ipmv.pt
wget https://raw.githubusercontent.com/bytedance/ImageDream/main/extern/ImageDream/imagedream/configs/sd_v2_base_ipmv.yaml
cd ..

# convert
python convert_mvdream_to_diffusers.py --checkpoint_path models/sd-v2.1-base-4view-ipmv.pt --dump_path ./weights_imagedream --original_config_file models/sd_v2_base_ipmv.yaml --half --to_safetensors --test
```

### Acknowledgement

* The original papers:
    ```bibtex
    @article{shi2023MVDream,
        author = {Shi, Yichun and Wang, Peng and Ye, Jianglong and Mai, Long and Li, Kejie and Yang, Xiao},
        title = {MVDream: Multi-view Diffusion for 3D Generation},
        journal = {arXiv:2308.16512},
        year = {2023},
    }
    @article{wang2023imagedream,
        title={ImageDream: Image-Prompt Multi-view Diffusion for 3D Generation},
        author={Wang, Peng and Shi, Yichun},
        journal={arXiv preprint arXiv:2312.02201},
        year={2023}
    }
    ```
* This codebase is modified from [mvdream-hf](https://github.com/KokeCacao/mvdream-hf).