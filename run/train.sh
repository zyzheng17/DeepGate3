nohup python3 -u src/train_dg3.py \
 --data_dir ./data/train_dg3 \
 --circuit_path ./data/train_dg3/graphs.npz \
 --pretrained_model_path ./trained/model_last.pth \
 >> /home/zyzheng23/project/DeepGate3-Transformer/data/data.log 2>&1 &

#  python3 -u src/train_dg3.py \
#  --data_dir ./data/winhop \
#  --circuit_path ./data/winhop/graphs_winhop.npz \
#  --pretrained_model_path ./trained/model_last.pth