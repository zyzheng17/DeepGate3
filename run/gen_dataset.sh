#!/bin/bash
python3 ./src/gen_dataset.py \
 --exp_id default \
 --gpus -1 \
 --data_dir ./data/dg3_all \
 --circuit_path ./data/dg3_all/graphs.npz 
