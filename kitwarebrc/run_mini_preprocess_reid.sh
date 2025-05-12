#!/bin/bash

#SBATCH --job-name=mini_reid
#SBATCH --mem=24GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=5-00:00:00
#SBATCH --output=mini_preprocess_reid.out

ps x
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

source /home/mauricio.segundo/.bashrc

conda activate reid
export PYTHONPATH="./:$PYTHONPATH"
python mini_preprocess_reid.py
conda deactivate



