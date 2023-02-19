#/bin/bash -l

python training.py \
--image_dir ../data/3090ti/rgb \
--labels_path ../data/3090ti/metrics/metrics.csv \
--save_path ../models/TestModel.pt \
--tb_path ../tb_logs/TestRun \
--use_gpus 0 \
--image_size 96 96 \
--batch_size 512 \
--num_workers 20 \
--epochs 100 \
--learning_rate 0.001 \
--weight_decay 0.0001
