#!/bin/bash
NUM_PROC=4
GPUS=0,1,2,3
DATA_RATIO=1

python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=29502 ./src/train_dg3.py \
 --exp_id train_${DATA_RATIO}p \
 --data_dir /home/zyshi21/data/inmemory/dg3_train_${DATA_RATIO}p \
 --npz_dir /home/zyshi21/data/share/dg3_dataset/${DATA_RATIO}p \
 --pretrained_model_path ./trained/model_last_workload.pth \
 --workload --gpus ${GPUS} --batch_size 128
