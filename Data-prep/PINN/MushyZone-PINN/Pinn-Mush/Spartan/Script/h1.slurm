#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Fri Aug 30 2024 17:20:44 GMT+1000 (Australian Eastern Standard Time)

# Partition for the job:
#SBATCH --partition=gpu-a100

# The name of the job:
#SBATCH --job-name="p1"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1

# Number of GPUs requested per node:
#SBATCH --gres=gpu:2
# The amount of memory in megabytes per cpu core:
#SBATCH --mem-per-cpu=65536

# Use this email address:
#SBATCH --mail-user=rrammohan@student.unimelb.edu.au

# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# begins
#SBATCH --mail-type=BEGIN
# ends successfully
#SBATCH --mail-type=END

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-7:0:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The modules to load:
module load Anaconda3/2022.10
module load eval "$(conda shell.bash hook)"
module load conda activate pytorch-env
module load cuDNN/8.9.3.28-CUDA-12.2.0

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
