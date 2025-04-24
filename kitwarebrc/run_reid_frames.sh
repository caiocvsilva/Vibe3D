#!/bin/bash

#SBATCH --job-name=reid_frames
#SBATCH --mem=30GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=5-00:00:00

ps x
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

source /home/mauricio.segundo/.bashrc

conda activate reid
export PYTHONPATH="./:$PYTHONPATH"
python reid_frames.py
conda deactivate


