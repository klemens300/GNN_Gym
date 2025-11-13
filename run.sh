#!/bin/sh
#--- parameters
#SBATCH -A imw
#SBATCH -J Active_Learning_GNN
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-freq=high
#SBATCH --exclusive
#SBATCH -p p2gpu
#SBATCH --qos p2gpu
#SBATCH --time 7-0:0:0
#SBATCH -o /mulfs/home/p2467946/aaa_New/GNN_Gym/slurm_output_%j.out
#SBATCH -e /mulfs/home/p2467946/aaa_New/GNN_Gym/slurm_error_%j.err
#---
ulimit -s unlimited

# Change to working directory
cd /mulfs/home/p2467946/aaa_New/GNN_Gym

# Activate conda/mamba
source /mulfs/home/p2467946/mambaforge/etc/profile.d/conda.sh
conda activate wandb

# Run the active learning script
srun -n1 -c16 -G1 python xxx_active_learning_loop_xxx.py