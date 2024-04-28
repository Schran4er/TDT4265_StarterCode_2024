#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --job-name="segres"
#SBATCH --output=out_%j.out



cd ${SLURM_SUBMIT_DIR}/

# module purge
# module load Anaconda3/2023.09-0
# conda activate tdt4265

CUDA_VISIBLE_DEVICES=0 python scripts/train.py run --config_file=configs/hyper_parameters.yaml

