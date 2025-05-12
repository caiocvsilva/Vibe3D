#!/bin/bash

#SBATCH --job-name=analyze
#SBATCH --mem=16GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00
#SBATCH --output=mapping_copy.out

ps x

source /home/mauricio.segundo/.bashrc

conda activate reid
export PYTHONPATH="./:$PYTHONPATH"

# Run the analysis script. Replace '/path/to/root' with your dataset path.
./mapping_copy.sh /blue/sarkar.sudeep/caio.dasilva/datasets/extracted_brc2 /blue/sarkar.sudeep/caio.dasilva/datasets/extracted_brc2_num 

conda deactivate