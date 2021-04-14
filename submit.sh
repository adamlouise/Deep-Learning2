#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=res.txt
#
#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --mem-per-cpu=100

module load numba/0.37.0-intel-2018a-Python-3.6.4

srun python3 Network2.py