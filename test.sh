#!/bin/sh
#SBATCH -A imw
#SBATCH -J Test_Oracle_Stress
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-freq=high
#SBATCH --exclusive
#SBATCH -p p2gpu
#SBATCH --qos p2gpu
#SBATCH --time 0-4:0:0
#SBATCH -o slurm_test_stress_%j.out
#SBATCH -e slurm_test_stress_%j.err

ulimit -s unlimited
cd $SLURM_SUBMIT_DIR

source /mulfs/home/p2467946/mambaforge/etc/profile.d/conda.sh
conda activate wandb-stable

echo "Running stress test with 3 calculations..."
srun -n1 -c16 -G1 python test_oracle.py --mode stress --calculator fairchem --n-tests 3