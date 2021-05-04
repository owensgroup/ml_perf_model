# ml_perf_model
A continuation of Zhongyi's ML performance model work at Facebook.

To get started, simply run
```bash
pip install h5py tqdm sklearn tensorboard torchviz onnx
git clone --recursive https://github.com/owensgroup/ml_perf_model.git
cd ml_perf_model/bench_params
./generate_benchmark_parameters.sh # Generate benchmark parameters (please modify the GPU memory size in the script).
cd ../sparse-ads-baselines
python setup.py install # Install table batched embedding lookup kernel.
cd ../mlperf-logging
python setup.py install # Install MLPerf logging for DLRM
cd ..
source ./init_vars.sh # Turn off turbo, turn on performance, lock frequency, etc.
```

The following benchmark commands are supported:
```bash
./benchmark.sh fully_connected 1 # 1 for forward
./benchmark.sh fully_connected 1 1 # The second 1 for big batch size.
./benchmark.sh cat 1
./benchmark.sh memcpy 1
./benchmark.sh transpose 1
./benchmark.sh embedding_lookup 1
./benchmark.sh embedding_lookup 0
./benchmark.sh embedding_lookup 1 1 # The second 1 for big batch size.
./benchmark.sh embedding_lookup 0 1
./benchmark.sh tril 1
./benchmark.sh tril 0
```

Notice: This code also depends on the private `facebookexternal/ml_perf_model` repo.

To train ML-based performance model for FC, transpose, and tril, run:
```bash
cd analysis/ml_predictors
python mlp.py --op-type fully_connected --batch-size 64
python mlp.py --op-type transpose --batch-size 32
python mlp.py --op-type tril --epoch 1000
python mlp.py --op-type tril --backward --epoch 2000
```