#!/bin/bash

#SBATCH --job-name=analyze
#SBATCH --mem=16GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00
#SBATCH --output=dataset_analysis.out

ps x

source /home/mauricio.segundo/.bashrc

conda activate reid
export PYTHONPATH="./:$PYTHONPATH"

# Run the analysis script. Replace '/path/to/root' with your dataset path.
python analyze_dataset.py /home/caio.dasilva/datasets/extracted_brc2 --save-prefix analysis_results

conda deactivate