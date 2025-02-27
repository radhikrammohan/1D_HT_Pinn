#!/bin/bash
#SBATCH --job-name=test-job    # Name of the job
#SBATCH --output=output_%j.txt        # Output file for logs
#SBATCH --error=error_%j.txt          # Error log file
#SBATCH -p feit-gpu-a100
#SBATCH --gres=gpu:3
#SBATCH --ntasks=1                 # Number of tasks (like chefs working on a dish)
#SBATCH --cpus-per-task=2          # Number of CPU cores per task (like each chef’s hands)
#SBATCH --mem=10G                   # Memory needed for the job (like ingredients required)
#SBATCH --time=00:30:00            # Max runtime (like how long your table is reserved)

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
python 1d-pinn-hpc.py --job_id $SLURM_JOB_ID 