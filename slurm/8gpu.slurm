#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBTACH --job-name=$config                              #jobname
#SBATCH --nodes=1                                       #This needs to match Trainer(num_nodes=...)
#SBTACH -p gpu                                          #partition
#SBATCH --gres=gpu:8                                    #gpus per node
#SBATCH --ntasks-per-node=8                             #This needs to match Trainer(devices=...)
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm/log/%x-%j.out.txt
#SBATCH --error=slurm/log/%x-%j.error.txt

config=$1
num_nodes=$2

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
echo "NODES: ${nodes[@]}"
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" ifconfig ibs22f1 | grep "inet " | awk '{print $2}')

export WANDB_API_KEY=""
export WANDB_MODE="offline"
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="ibs22f1"
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export MASTER_PORT=12345
export MASTER_ADDR=$head_node_ip
export WORK_DIR=/mnt/cfs/home/liweiyu/codes/CraftsMan
export PYTHONPATH=/mnt/cfs/home/liweiyu/.conda/envs/lrm/bin/python3
export CUDA_HOME=/mnt/cfs/apps/cuda-12.2

module load cuda/12.2
source activate lrm

echo "SLURM_JOBID: " $SLURM_JOBID

# run script from above
srun python3 $WORK_DIR/launch.py \
    --train \
    --gpu 0,1,2,3,4,5,6,7 \
    --config $WORK_DIR/configs/$config \
    trainer.num_nodes=$num_nodes
