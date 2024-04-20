#!/bin/bash
NUM_PROC=8
GPUS=0,1,2,3,4,5,6,7

# nohup  python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=29547 ./DeepGate3-Transformer/src/train_dg3.py \
nohup python3 -u -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=29517 ./DeepGate3-Transformer/src/train_dg3.py \
 --exp_id baseline_full \
 --gpus ${GPUS} \
 --data_dir ./DeepGate3-Transformer/data/plain_10k \
 --load_npz ./DeepGate3-Transformer/data/plain_10k/10k_wl_4_hop.npz \
 --pretrained_model_path ./DeepGate3-Transformer/trained/model_last_workload.pth \
 --tf_arch baseline --workload \
 --batch_size 56 --lr 5e-4 \
 >> /home/zyzheng23/project/DeepGate3-Transformer/exp/0417_baseline_workload_8k.log 2>&1 &