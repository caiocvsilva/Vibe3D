#!/bin/bash

#SBATCH --job-name=brc_data
#SBATCH --mem=16GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus=a100:2
#SBATCH --time=5-00:00:00

ps x
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

source /home/mauricio.segundo/.bashrc

conda activate pytorch3d
export PYTHONPATH="./:$PYTHONPATH"
python lib/data_utils/brc2_utils.py
conda deactivate

