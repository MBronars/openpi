#!/bin/bash

#SBATCH --job-name=training # Job name
#SBATCH --output=/home/mbronars/workspace/slurm_runs/3DVLA/rlbench_output_%j.log
#SBATCH --error=/home/mbronars/workspace/slurm_runs/3DVLA/rlbench_error_%j.log
#SBATCH --time=2-00:00:00             # Time limit (hh:mm:ss)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks (processes)
#SBATCH --gpus-per-task=1          # Number of tasks (processes)
#SBATCH --constraint="A6000|6000Ada|L40|L40S"
#SBATCH --cpus-per-task=6          # Number of cores per task
#SBATCH --mem=150G                  # Memory limit per node
#SBATCH --partition=general# Partition name

cd /data/user_data/mbronars/packages/openpi

echo "Launching pi0 server for peract2 evaluation"
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_peract2 --policy.dir=checkpoints/pi0_peract2/peract2_test/29999 &


source /home/mbronars/.bashrc
source /home/mbronars/miniconda3/etc/profile.d/conda.sh
conda activate robot_26

echo "Launching evaluation script"
cd /data/user_data/mbronars/packages/analogical_manipulation
bash /data/user_data/mbronars/packages/analogical_manipulation/online_evaluation_rlbench/eval_peract2.sh