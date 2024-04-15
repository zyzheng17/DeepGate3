#!/bin/bash
NUM_PROC=2
GPUS=0,1

nohup python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=29517 ./src/train_dg3.py \
 --exp_id plain_full \
 --gpus ${GPUS} \
 --data_dir ./data/train_dg3 \
 --circuit_path ./data/train_dg3/graphs.npz \
 --pretrained_model_path ./trained/model_last.pth \
 --tf_arch plain \
 --batch_size 8 --fast \
 >> /uac/gds/zyzheng23/projects/DeepGate3-Transformer/exp/0414_plain.log 2>&1 &