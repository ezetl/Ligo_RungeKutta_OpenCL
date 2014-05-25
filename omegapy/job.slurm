#!/bin/bash
### Job settings
#SBATCH --job-name tinyIsingRB
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SALLOC --gres=gpu:1
#SBATCH --time=00:10:00

### Environment setup
source /etc/profile
## module load compilers/gcc/4.8.2

### Run task
./main.py
