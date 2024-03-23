#!/bin/bash

EXP_ID=test_hop
ARCH=hop

python ./src/test_dg3.py \
 --exp_id $EXP_ID \
 --gpus -1 --batch_size 1 \
 --data_dir ./data/large --enable_large_circuit \
 --circuit_path ./data/large/graphs.npz \
 --pretrained_model_path ./trained/model_last.pth \
 --tf_arch $ARCH \
 --no_cone --test --no_stru
