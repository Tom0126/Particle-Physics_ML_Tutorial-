#! /bin/bash


######## Part 1 #########
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --account=higgsgpu
#SBATCH --ntasks=5
#SBATCH --mem-per-cpu=16GB
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=pid


######## Part 2 ######
# Script workload    #
######################

# Replace the following lines with your real workload

source /hpcfs/cepc/higgsgpu/siyuansong/conda.env
conda activate pytorch




python /hpcfs/cepc/higgsgpu/siyuansong/PID/Model/ANA/e_sigma.py


