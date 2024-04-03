#!/bin/bash
NUM_PROC=2
GPUS=0,1

python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=29500 ./src/train_dg3_cl.py \
 --exp_id cl_winhop \
 --gpus ${GPUS} \
 --data_dir ./data/winhop \
 --load_npz ./data/winhop/graphs_winhop.npz \
 --pretrained_model_path ./trained/model_last.pth \
 --tf_arch hop --TF_depth 6 \
 --batch_size 16 --fast