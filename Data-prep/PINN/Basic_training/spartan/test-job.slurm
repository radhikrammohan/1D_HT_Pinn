#!/bin/bash
#SBATCH --job-name=test-job    # Name of the job
#SBATCH --output=output.txt        # Output file for logs
#SBATCH --error=error.txt          # Error log file
#SBATCH --ntasks=1                 # Number of tasks (like chefs working on a dish)
#SBATCH --cpus-per-task=4          # Number of CPU cores per task (like each chef’s hands)
#SBATCH --mem=8G                   # Memory needed for the job (like ingredients required)
#SBATCH --time=00:30:00            # Max runtime (like how long your table is reserved)
#SBATCH --partition=standard       # Partition/queue to run on (like choosing a fast-food or fine-dining section)
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1

# Load necessary modules
module load python/3.10.4           # Load Python module
module load PyTorch/2.1.2-CUDA-12.2.0           # Load PyTorch (check with `module avail pytorch`)


# (Optional) Create or activate a virtual environment
source ~/pytorch-env/bin/activate

# Print environment details (optional)
echo "Loaded Modules:"
module list

echo "Running on node: $(hostname)"
echo "CUDA devices available: $(nvidia-smi)"

# Run the Python script
python 1d-pinn-hpc.py --job_id $SLURM_JOB_ID 