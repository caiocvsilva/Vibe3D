#!/bin/bash

#SBATCH --job-name=reid_frames
#SBATCH --mem=24GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus=a100:2
#SBATCH --time=5-00:00:00
#SBATCH --output=preprocess_reid_gpu.out

ps x
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

source /home/mauricio.segundo/.bashrc

conda activate reid
export PYTHONPATH="./:$PYTHONPATH"
python preprocess_reid_gpu.py
conda deactivate



