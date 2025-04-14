# DeepGate3: Graph-level Circuit Representation Learning

Official code repository for the paper: DeepGate3: Graph-level Circuit Representation Learning

## Installation
For conda environments, please refer to https://github.com/zshi0616/DeepGate2

Install python-deepgate
```sh
git clone https://github.com/Ironprop-Stone/python-deepgate.git
cd python-deepgate
bash install.sh
```

Prepare required tools
```sh 
bash build.sh
```
# DeepGate3
To run the training stage of DeepGate3 with single GPU, you can run the following script:
```
 python -u  ./src/train_dg3.py \
 --exp_id you_exp_id \
 --data_dir path/to/inmemory \
 --npz_dir pat/to/npz \
 --pretrained_model_path \path\to\dg2_ckpt \
 --tf_arch plain --lr 1e-4  \
 --workload --gpus 0 --batch_size 128 --epoch 200
```
If there is no data in data_dir, the code will parse data from npz_dir and generate inmemory data in data_dir.
