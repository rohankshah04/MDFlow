#!/bin/bash
#SBATCH --job-name=msa_info         # Job name
#SBATCH --output=output_%j.txt        # Output file, %j will be replaced with the job ID
#SBATCH --error=error_%j.txt          # Error file, %j will be replaced with the job ID
#SBATCH --ntasks=1                    # Number of tasks (processes)
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem=16G                     # Memory per node
#SBATCH --time=04:00:00               # Time limit hrs:min:sec
#SBATCH --partition=all               # Partition name

# Load the necessary modules (if any)
# module load anaconda

# Activate Conda environment
conda init bash
source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh  # Ensure conda command is available
conda activate shah_mdflow

# Run your Python script
python add_msa_info.py 
~                                  