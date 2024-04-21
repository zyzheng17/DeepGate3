NUM_PROC=8
GPUS=0,1,2,3,4,5,6,7
DATA_RATIO=10


# NUM_PROC=1
# GPUS=0
# DATA_RATIO=50

nohup python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=29557 ./DeepGate3-Transformer/src/train_dg3.py \
 --exp_id train_${DATA_RATIO}p \
 --data_dir /home/zyshi21/data/inmemory/dg3_train_${DATA_RATIO}p \
 --npz_dir /home/zyshi21/data/share/dg3_dataset/${DATA_RATIO}p \
 --pretrained_model_path ./DeepGate3-Transformer/trained/model_last_workload.pth \
 --tf_arch plain --lr 1e-4 \
 --workload --gpus ${GPUS} --batch_size 128 \
 >> /home/zyzheng23/project/DeepGate3-Transformer/exp/0420_plain_workload_${DATA_RATIO}p_balanced_unfixed.log 2>&1 &
