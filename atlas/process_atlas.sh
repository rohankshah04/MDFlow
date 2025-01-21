#!/bin/bash
#SBATCH --job-name=mdflow_data          # Job name
#SBATCH --output=output_%j.txt        # Output file, %j will be replaced with the job ID
#SBATCH --error=error_%j.txt          # Error file, %j will be replaced with the job ID
#SBATCH --ntasks=1                  # Number of tasks (processes)
#SBATCH --cpus-per-task=50           # Number of CPU cores per task
#SBATCH --mem=50G                     # Memory per node
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --partition=all               # Partition name

# -----------------------------------
# Activate Conda environment
conda init bash
source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh  # Ensure conda command is available
conda activate shah_mdflow

# Define S3 bucket name
S3_BUCKET="mdflow.atlas"

# Download and process each dataset
for name in $(cat /cbica/home/shahroha/projects/AF-DIT/atlas/atlas_part20.csv | grep -v name | awk -F ',' '{print $1}'); do
    wget https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/${name}/${name}_protein.zip
    mkdir -p ${name}
    unzip ${name}_protein.zip -d ${name}
    python -m prep_atlas \
        --atlas_dir ${name} \
        --outdir ${name}_processed \
        --num_workers 1
    aws s3 cp ${name}_processed s3://${S3_BUCKET}/${name} --recursive
    rm -rf ${name} ${name}_protein.zip ${name}_processed
done