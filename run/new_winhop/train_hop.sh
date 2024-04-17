#!/bin/bash
NUM_PROC=8
GPUS=0,1,2,3,4,5,6,7

# nohup python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=29501 ./DeepGate3-Transformer/src/train_dg3.py \
#  --exp_id cl_winhop \
#  --gpus ${GPUS} \
#  --data_dir ./DeepGate3-Transformer/data/winhop \
#  --pretrained_model_path ./DeepGate3-Transformer/trained/model_last.pth \
#  --tf_arch hop --TF_depth 12 \
#  --batch_size 8 --fast \
#   --load_npz ./DeepGate3-Transformer/data/winhop/graphs_winhop.npz \
#   >> ./DeepGate3-Transformer/exp/0416_hop_nocl.log 2>&1 &

python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=29501 ./DeepGate3-Transformer/src/train_dg3.py \
 --exp_id cl_winhop \
 --gpus ${GPUS} \
 --data_dir ./DeepGate3-Transformer/data/winhop \
 --pretrained_model_path ./DeepGate3-Transformer/trained/model_last.pth \
 --tf_arch hop --TF_depth 12 \
 --batch_size 16 \
  --load_npz ./DeepGate3-Transformer/data/winhop/graphs_winhop.npz
