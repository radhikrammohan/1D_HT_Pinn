#!/bin/bash

#SBATCH --account="punim2143"  # Specify the account to be charged for the job
#SBATCH --job-name="testjob"   # Name of the job, which will appear in the SLURM queue


# Optional partition and QoS settings for deep learning, commented out for now

# SBATCH --partition=deeplearn  # Partition for deep learning jobs (commented out)
# SBATCH --qos=gpgpudeeplearn   # QoS for deep learning jobs (commented out)
# SBATCH --time=1-0:00:00       # Time limit for the job (1 day), commented out

#SBATCH --partition=gpgpu       # Specify the partition to submit to (gpgpu partition)
#SBATCH --qos=gpgpumse          # Specify the QoS (Quality of Service) for this job (gpgpumse)
#SBATCH --time=1-0:00:00         # Time limit for the job (1 day)

#SBATCH --gres=gpu:1              # Request 1 GPU for the job

# Specify output and error file locations
#SBATCH --output=/data/gpfs/projects/punimxxxx/test_folder/job_output_%j.out  # Standard output (%j represents job ID)
#SBATCH --error=/data/gpfs/projects/punimxxxx/test_folder/job_error_%j.err    # Standard error (%j represents job ID)


# load necessary modules #

cd /data/gpfs/projects/punimxxxx/test_folder/

# Load the necessary modules for the job environment

# Remove all previously loaded modules (optional, commented out)

# module purge
module load fosscuda/2020b
module load gcc/10.2.0 openmpi/4.0.5 cuda/11.1.1 
module load python/3.8.6
module load pytorch/1.9.0-python-3.8.6
module load tensorflow/2.6.0-python-3.8.6
module load anaconda3/2021.05

# Run the Python script with unbuffered output
python -u train.py   # First job
