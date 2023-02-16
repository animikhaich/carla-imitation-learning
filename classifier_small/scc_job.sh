#!/bin/bash -l

# Request 16 CPUs
#$ -pe omp 16

# Request 1 GPU 
#$ -l gpus=1

# Specify the minimum GPU compute capability. 
#$ -l gpu_c=6.0

# Email when done
#$ -m ea

# Combine output and error files into a single file
#$ -j y

# 42 hours
#$ -l h_rt=42:00:00

module load miniconda
conda activate rlvn
module load pytorch
python training.py -d ../data/combined/rgb -l ../data/combined/metrics.csv 
