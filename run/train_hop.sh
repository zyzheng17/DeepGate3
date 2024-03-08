python3 ./src/train_dg3.py \
 --exp_id hop \
 --data_dir ./data/dg3_all \
 --circuit_path ./data/dg3_all/graphs.npz \
 --pretrained_model_path ./trained/model_last.pth \
 --tf_arch hop \
 --batch_size 8 \
 --w_prob 1.0 --w_tt_sim 0 --w_tt_cls 1.0 --w_g_sim 0 \
 --no_cone
