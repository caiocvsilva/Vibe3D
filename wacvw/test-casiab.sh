#!/bin/bash

#SBATCH --job-name=mytrain
#SBATCH --mem=15GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=5-00:00:00

ps x
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

source /home/mauricio.segundo/.bashrc

conda activate pytorch3d
export PYTHONPATH="./:$PYTHONPATH"
python test-casiab.py
conda deactivate

