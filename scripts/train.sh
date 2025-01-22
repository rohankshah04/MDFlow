#!/bin/bash
#SBATCH --job-name=train_mdflow         # Job name
#SBATCH --output=output_%j.txt        # Output file, %j will be replaced with the job ID
#SBATCH --error=error_%j.txt   
#SBATCH --nodes=1 # Error file, %j will be replaced with the job ID
#SBATCH --ntasks=1                    # Number of tasks (processes)
#SBATCH --cpus-per-task=4    
#SBATCH --gpus-per-node=a100:1         # Number of CPU cores per task
#SBATCH --mem=32G                     # Memory per node
#SBATCH --time=20:00:00               # Time limit hrs:min:sec
#SBATCH --partition=all               # Partition name

ulimit -n 65536
unlimit -s unlimited

# Load the necessary modules (if any)
# module load anaconda

# Activate Conda environment
conda init bash
source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh  # Ensure conda command is available
conda activate shah_mdflow

# export OMP_NUM_THREADS=1

export WANDB_API_KEY="8ac276ce003c8b51424e5c554ad85d4beb7cdbaa"
export WANDB_PROJECT="mdflow"
export WANDB_ENTITY="mdflow"
wandb login ${WANDB_API_KEY}

PROJECT_DIR="/cbica/home/shahroha/projects/AF-DIT"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Print debug information
echo "Current directory: $PWD"
echo "Project directory: $PROJECT_DIR"
echo "PYTHONPATH: $PYTHONPATH"

# Run your Python script
python $PROJECT_DIR/mdflow/model/train.py \
    --overfit_samples 2 \
    --epochs 100 \
    --batch_size 1 \
    --lr 1e-4 \
    --run_name mdflow_overfit_test \
    --wandb_project mdflow \
    --noise_prob 0.8 \
    --val_freq 1 \
    --ckpt_freq 5
~