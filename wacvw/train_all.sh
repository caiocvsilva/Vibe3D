#!/bin/bash

#SBATCH --job-name=train_all
#SBATCH --mem=30GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus=a100:2
#SBATCH --time=5-00:00:00
#SBATCH --output=brc2_train.out

ps x
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

source /home/mauricio.segundo/.bashrc

conda activate pytorch3d
export PYTHONPATH="./:$PYTHONPATH"
python train_all.py
conda deactivate

