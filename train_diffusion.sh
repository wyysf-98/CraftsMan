export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

python launch.py --config ./configs/image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6.yaml --train --gpu 0