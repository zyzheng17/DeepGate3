#!/bin/bash
NUM_PROC=4
GPUS=0,1,2,3

python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC ./src/train_dg3.py \
 --exp_id plain \
 --gpus ${GPUS} \
 --data_dir ./data/dg3_all \
 --circuit_path ./data/dg3_all/graphs.npz \
 --pretrained_model_path ./trained/model_last.pth \
 --tf_arch plain \
 --batch_size 8 