#!/bin/bash
NUM_PROC=8
GPUS=0,1,2,3,4,5,6,7

# nohup python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=29517 ./DeepGate3-Transformer/src/train_dg3.py \
#  --exp_id plain_full \
#  --gpus ${GPUS} \
#  --data_dir ./DeepGate3-Transformer/data/train_dg3 \
#  --circuit_path ./DeepGate3-Transformer/data/train_dg3/graphs.npz \
#  --pretrained_model_path ./DeepGate3-Transformer/trained/model_last.pth \
#  --tf_arch plain --workload \
#  --batch_size 16 --fast \
#  >> ./DeepGate3-Transformer/exp/0416_plain.log 2>&1 &

python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=29517 ./DeepGate3-Transformer/src/train_dg3.py \
 --exp_id plain_full \
 --gpus ${GPUS} \
 --data_dir ./DeepGate3-Transformer/data/train_dg3 \
 --circuit_path ./DeepGate3-Transformer/data/train_dg3/graphs.npz \
 --pretrained_model_path ./DeepGate3-Transformer/trained/model_last.pth \
 --tf_arch plain --workload \
 --batch_size 16 --fast \
