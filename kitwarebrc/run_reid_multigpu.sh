#!/bin/bash

#SBATCH --job-name=reid_frames
#SBATCH --mem=60GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=a100:3
#SBATCH --time=5-00:00:00
#SBATCH --output=reid_frames_%j.out
#SBATCH --error=reid_frames_%j.err

ps x
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

source /home/mauricio.segundo/.bashrc

conda activate reid
export PYTHONPATH="./:$PYTHONPATH"
python reid_multi_gpu.py
conda deactivate


