nohup python3 -u src/train_dg3.py \
 --data_dir ./data/dg3_all \
 --circuit_path ./data/dg3_all/graphs.npz \
 --pretrained_model_path ./trained/model_last.pth >> /uac/gds/zyzheng23/projects/DeepGate3-Transformer/data/data.log 2>&1 &