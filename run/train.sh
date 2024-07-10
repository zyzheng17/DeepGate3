# NUM_PROC=6
# GPUS=0,1,2,3,4,5
# DATA_RATIO=10

# NUM_PROC=1
# GPUS=2
# DATA_RATIO=10

# nohup python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=29557 ./DeepGate3-ICCAD/src/train_dg3_cl.py \
#  --exp_id train_plain_${DATA_RATIO}p_cl \
#  --data_dir /home/zyzheng23/project/data/inmemory/dg3_train_${DATA_RATIO}p \
#  --npz_dir /home/zyzheng23/project/dg3_dataset/${DATA_RATIO}p \
#  --pretrained_model_path ./DeepGate3-ICCAD/trained/model_last_workload.pth \
#  --tf_arch plain --lr 1e-4 \
#  --workload --gpus ${GPUS} --batch_size 128 \
#  >> /home/zyzheng23/project/DeepGate3-ICCAD/exp/0422_plain_cl_workload_${DATA_RATIO}p_balanced_unfixed.log 2>&1 &


# NUM_PROC=4
# GPUS=0,1,2,3
# DATA_RATIO=10
# MODE=baseline
# DATE=0628
# COND='128x4_4_2'


# nohup python -u -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=29527 ./DeepGate3-ICCAD/src/train_dg3.py \
#  --exp_id ${DATE}_train_${MODE}_${DATA_RATIO}p_${COND} \
#  --data_dir /home/zyzheng23/project/data/inmemory/dg3_train_${DATA_RATIO}p \
#  --npz_dir /home/zyzheng23/project/dg3_dataset/${DATA_RATIO}p \
#  --pretrained_model_path ./DeepGate3-ICCAD/trained/model_last_workload.pth \
#  --tf_arch ${MODE} --lr 1e-4 --en_distrubuted \
#  --workload --gpus ${GPUS} --batch_size 128 --epoch 200 \
#  >> /home/zyzheng23/project/DeepGate3-ICCAD/exp/${DATE}_${MODE}_${DATA_RATIO}p_${COND}.log 2>&1 &

# NUM_PROC=0
# GPUS=2
NUM_PROC=4
GPUS=4,5,6,3
DATA_RATIO=50
MODE=baseline
DATE=0710
COND='12_4'


# python  ./DeepGate3-ICCAD/src/train_dg3.py \
#  --exp_id ${DATE}_train_${MODE}_${DATA_RATIO}p_${COND} \
#  --data_dir /home/zyzheng23/project/data/inmemory/dg3_train_${DATA_RATIO}p \
#  --npz_dir /home/zyzheng23/project/dg3_dataset/${DATA_RATIO}p \
#  --pretrained_model_path ./DeepGate3-ICCAD/trained/model_last_workload.pth \
#  --tf_arch ${MODE} --lr 1e-4 --en_distrubuted \
#  --workload --gpus ${GPUS} --batch_size 128 --epoch 200 

nohup python -u -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=29593 ./DeepGate3-ICCAD/src/train_dg3.py \
 --exp_id ${DATE}_train_${MODE}_${DATA_RATIO}p_${COND} \
 --data_dir /home/zyzheng23/project/data/inmemory/dg3_train_${DATA_RATIO}p \
 --npz_dir /home/zyzheng23/project/dg3_dataset/${DATA_RATIO}p \
 --pretrained_model_path ./DeepGate3-ICCAD/trained/model_last_workload.pth \
 --tf_arch ${MODE} --lr 1e-4 --en_distrubuted \
 --workload --gpus ${GPUS} --batch_size 128 --epoch 200 \
 >> /home/zyzheng23/project/DeepGate3-ICCAD/exp/${DATE}_${MODE}_${DATA_RATIO}p_${COND}.log 2>&1 &


