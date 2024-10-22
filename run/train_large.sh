#!/bin/bash

# NUM_PROC=8
# GPUS=0,1,2,3,4,5,6,7

NUM_PROC=0
GPUS=1

# nohup python -u -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=29557  ./DeepGate3-Transformer/src/train_dg3.py \
#  --exp_id train_large \
#  --data_dir /home/zyzheng23/project/data/inmemory/train_large \
#  --npz_dir /home/zyzheng23/project/dg3_dataset/large_train \
#  --pretrained_model_path ./DeepGate3-Transformer/trained/model_last_workload.pth \
#  --tf_arch plain --lr 1e-4 --enable_cut \
#  --workload --gpus ${GPUS} --batch_size 1 \
#  >> /home/zyzheng23/project/DeepGate3-Transformer/exp/0504_train_large_avg.log 2>&1 &
# 

nohup python -u ./DeepGate3-Transformer/src/train_dg3.py \
 --exp_id train_large \
 --data_dir /home/zyzheng23/project/data/inmemory/train_large \
 --npz_dir /home/zyzheng23/project/dg3_dataset/large_train \
 --pretrained_model_path ./DeepGate3-Transformer/trained/model_last_workload.pth \
 --tf_arch plain --lr 1e-4 --enable_cut \
 --workload --gpus ${GPUS} --batch_size 1 \
 >> /home/zyzheng23/project/DeepGate3-Transformer/exp/0504_train_large_avg.log 2>&1 &

# nohup python -u ./DeepGate3-Transformer/src/train_dg3.py \
#  --exp_id train_large \
#  --data_dir /home/zyzheng23/project/data/inmemory/train_large \
#  --npz_dir /home/zyzheng23/project/dg3_dataset/large_train \
#  --pretrained_model_path ./DeepGate3-Transformer/trained/model_last_workload.pth \
#  --tf_arch baseline --lr 1e-4 --enable_cut \
#  --workload --gpus ${GPUS} --batch_size 1 -skip_hop \
#  >> /home/zyzheng23/project/DeepGate3-Transformer/exp/0503_train_large_baseline.log 2>&1 &