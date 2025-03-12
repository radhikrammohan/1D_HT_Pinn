#!/bin/bash
#SBATCH --job-name=test-job    # Name of the job
#SBATCH --output=/home/rrammohan/jobs/1D_HT_Pinn/Data-prep/PINN/Basic_training/spartan/output_files/job_%j/output_%j.txt        # Output file for logs
#SBATCH --error=/home/rrammohan/jobs/1D_HT_Pinn/Data-prep/PINN/Basic_training/spartan/output_files/job_%j/error_%j.txt          # Error log file
#SBATCH --partition=gpu-h100 
#SBATCH --gres=gpu:2

#SBATCH --ntasks=1                 # Number of tasks (like chefs working on a dish)
#SBATCH --cpus-per-task=2          # Number of CPU cores per task (like each chef’s hands)
#SBATCH --mem=10G                   # Memory needed for the job (like ingredients required)
#SBATCH --time=2-00:00:00            # Max runtime (like how long your table is reserved)

#SBATCH --qos=feit

# Load necessary modules
module purge
module load Anaconda3
source ~/.bashrc
conda activate pytorch-env





# Print environment details (optional)
echo "Loaded Modules:"
module list

echo "Running on node: $(hostname)"
echo "CUDA devices available: $(nvidia-smi)"

# Run the Python script
python 1d-normal.py --job_id $SLURM_JOB_ID 