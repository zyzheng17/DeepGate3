NUM_PROC=8
GPUS=0,1,2,3,4,5,6,7
DATA_RATIO=5

python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=29558 ./src/train_dg3.py \
 --exp_id train_${DATA_RATIO}p \
 --data_dir /home/zyshi21/data/inmemory/01_dg3_train_${DATA_RATIO}p \
 --npz_dir /home/zyshi21/data/share/dg3_dataset/${DATA_RATIO}p \
 --pretrained_model_path ./trained/model_last_workload.pth \
 --tf_arch plain --lr 1e-4 \
 --workload --gpus ${GPUS} --batch_size 128 \
 >> ./exp/0421_plain_workload_${DATA_RATIO}p_balanced_unfixed.log 