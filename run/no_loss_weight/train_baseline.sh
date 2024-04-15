#!/bin/bash
NUM_PROC=2
GPUS=3,4

 nohup  python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=29547 ./DeepGate3-Transformer/src/train_dg3.py \
 --exp_id baseline \
 --gpus ${GPUS} \
 --data_dir ./DeepGate3-Transformer/data/train_dg3 \
 --circuit_path ./DeepGate3-Transformer/data/train_dg3/graphs.npz \
 --pretrained_model_path ./DeepGate3-Transformer/trained/model_last.pth \
 --tf_arch baseline \
 --batch_size 8 --fast \
   >> /uac/gds/zyzheng23/projects/DeepGate3-Transformer/exp/0414_baseline.log 2>&1 &