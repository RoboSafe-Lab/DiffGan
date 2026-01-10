#!/bin/bash
#SBATCH --job-name=IRLTrain
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-24:00:00
#SBATCH --mem-per-cpu=100G
## GPU requirements
#SBATCH --gres gpu:1
#SBATCH -p gpu
#SBATCH --nice

export PYTHONUNBUFFERED=1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate bg3.9

# Change to the project directory
cd $HOME/DiffGAN

# Set PYTHONPATH to include both CTG and the current directory
export PYTHONPATH=$HOME/DiffGAN/CTG/:$HOME/DiffGAN:$PYTHONPATH

# Define parameter combinations

export WANDB_APIKEY=YOUR_API_KEY
# Run the appropriate command
python -m scripts.train --config=config/maxEntIRL.json

