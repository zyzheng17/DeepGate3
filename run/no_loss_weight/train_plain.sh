#!/bin/bash
NUM_PROC=2
GPUS=0,1

python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=29501 ./src/train_dg3.py \
 --exp_id plain_full \
 --gpus ${GPUS} \
 --data_dir ./data/dg3_all \
 --circuit_path ./data/dg3_all/graphs.npz \
 --pretrained_model_path ./trained/model_last.pth \
 --tf_arch plain \
 --batch_size 8 --fast 