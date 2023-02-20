#!/bin/bash -l

# Request 16 CPUs
#$ -pe omp 16

# Request 1 GPU 
#$ -l gpus=1

# Specify the minimum GPU compute capability. 
#$ -l gpu_c=7.0

# Email when done
#$ -m ea

# Combine output and error files into a single file
#$ -j y

# 42 hours
#$ -l h_rt=42:00:00

module load miniconda
conda activate rlvn
module load pytorchc

python training.py \
--image_dir ../data/combined/rgb \
--labels_path ../data/combined/metrics.csv \
--save_path ../models/small_regresson_v1.pt \
--tb_path ../tb_logs/small_regresson_v1 \
--use_gpus 0 \
--image_size 128 128 \
--batch_size 64 \
--num_workers 16 \
--epochs 500 \
--learning_rate 0.001 \
--weight_decay 1e-4
