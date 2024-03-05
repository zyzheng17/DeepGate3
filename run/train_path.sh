python3 -u ./src/train_dg3.py \
 --data_dir ./data/dg3_all \
 --circuit_path ./data/dg3_all/graphs.npz \
 --pretrained_model_path ./trained/model_last.pth \
 --tf_arch path \
 --sample_path_data --enable_large_circuit \
 >> ./exp/data.log 