#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --job-name="pytorch-lightning-demo"
#SBATCH --output=pytorch-lightning-demo.out



cd ${SLURM_SUBMIT_DIR}/

# module purge
# module load Anaconda3/2023.09-0
# conda activate tdt4265

srun python trainer_own.py
