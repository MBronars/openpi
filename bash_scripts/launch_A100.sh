#!/bin/bash

#SBATCH --job-name=training # Job name
#SBATCH --output=/home/mbronars/workspace/slurm_runs/3DVLA/rlbench_output_%j.log
#SBATCH --error=/home/mbronars/workspace/slurm_runs/3DVLA/rlbench_error_%j.log
#SBATCH --time=2-00:00:00             # Time limit (hh:mm:ss)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks (processes)
#SBATCH --gpus-per-task=1          # Number of tasks (processes)
#SBATCH --constraint="A100_80GB"
#SBATCH --cpus-per-task=6          # Number of cores per task
#SBATCH --mem=150G                  # Memory limit per node
#SBATCH --partition=general# Partition name

nvidia-smi
echo "GPU Topology:"
nvidia-smi topo -m

cd /data/user_data/mbronars/packages/openpi

bash /data/user_data/mbronars/packages/openpi/bash_scripts/train_mbronars.sh