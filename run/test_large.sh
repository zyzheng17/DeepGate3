#!/bin/bash

NUM_PROC=1
GPUS=0
DATA_RATIO=100


# nohup python ./DeepGate3-Transformer/src/train_dg3_cl.py \
#  --exp_id train_plain_${DATA_RATIO}p_cl \
#  --data_dir /home/zyzheng23/project/data/inmemory/dg3_train_${DATA_RATIO}p \
#  --npz_dir /home/zyzheng23/project/dg3_dataset/${DATA_RATIO}p \
#  --pretrained_model_path ./DeepGate3-Transformer/trained/model_last_workload.pth \
#  --tf_arch plain --lr 1e-4 \
#  --workload --gpus ${GPUS} --batch_size 1 \
#  >> /home/zyzheng23/project/DeepGate3-Transformer/exp/0422_plain_cl_workload_${DATA_RATIO}p_balanced_unfixed.log 2>&1 &

# python /home/zyzheng23/project/DeepGate3-Transformer/src/test_dg3_large.py \
#  --exp_id test_large \
#  --data_dir /home/zyzheng23/project/data/inmemory/large_test \
#  --npz_dir /home/zyzheng23/project/dg3_dataset/large_test \
#  --pretrained_model_path ./DeepGate3-Transformer/trained/model_last_workload.pth \
#  --tf_arch plain  --enable_cut \
#  --workload --gpus ${GPUS} --batch_size 1

nohup python -u ./DeepGate3-Transformer/src/train_dg3.py \
 --exp_id train_large \
 --data_dir /home/zyzheng23/project/data/inmemory/large_test \
 --npz_dir /home/zyzheng23/project/dg3_dataset/large_test \
 --pretrained_model_path ./DeepGate3-Transformer/trained/model_last_workload.pth \
 --tf_arch plain --lr 1e-4 --enable_cut \
 --workload --gpus ${GPUS} --batch_size 1 \
 >> /home/zyzheng23/project/DeepGate3-Transformer/exp/0425_train_large.log 2>&1 &
