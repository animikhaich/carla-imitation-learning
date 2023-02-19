#/bin/bash -l

python training.py \
--image_dir ../data/combined/rgb \
--labels_path ../data/combined/metrics.csv \
--save_path ../models/small_regresson_v1.pt \
--tb_path ../tb_logs/small_regresson_v1 \
--use_gpus 0 \
--image_size 128 128 \
--batch_size 64 \
--num_workers 20 \
--epochs 500 \
--learning_rate 0.1 \
--weight_decay 0.0
