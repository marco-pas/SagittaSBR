#!/bin/bash
#SBATCH -A NAISS2025-5-714
#SBATCH -p gpu
#SBATCH --qos=test           # Add this for short tests
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -t 00:01:00
#SBATCH -J SagittaTest

# Load the environment
module load rocm/6.3.3

# Enable GPU support for MPICH
export MPICH_GPU_SUPPORT_ENABLED=1

# Run the executable (total tasks = 8)
srun ./build/RT-RCS --model ./assets/test_models/a380.obj

