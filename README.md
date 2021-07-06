# dlrm_gpu_perf_model
DLRM GPU Training Performance Model

To get started, simply run
```bash
pip install h5py tqdm sklearn tensorboard torchviz onnx gputil # Dependencies
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
./benchmark.sh concat 1
./benchmark.sh memcpy 1
./benchmark.sh transpose 1
./benchmark.sh embedding_lookup 1
./benchmark.sh embedding_lookup 0
./benchmark.sh embedding_lookup 1 1 # The second 1 for big batch size.
./benchmark.sh embedding_lookup 0 1
./benchmark.sh tril 1
./benchmark.sh tril 0
./benchmark.sh conv 1 # We also support convolution for comparison with other performance models on DL models other the DLRM
./benchmark.sh conv 1 1
```

Notice: This code also depends on the private `facebookexternal/ml_perf_model` repo.

To train ML-based performance model for FC, conv, transpose, and tril, run:
```bash
python mlp.py --op-type fully_connected --batch-size 64
python mlp.py --op-type conv --batch-size 64
python mlp.py --op-type transpose --batch-size 32
python mlp.py --op-type tril --epoch 1000
python mlp.py --op-type tril --backward --epoch 2000
```

To print all performance model error rates, run:
```bash
python kernel_pm_acc.py
```

To extract execution graph and generate profiler trace file, run:
```bash
./dlrm_s_benchmark_local.sh <model_name> # Default is MLPerf
```

To generate trace file stats and overheads, and conduct end-to-end performance prediction, run:
```bash
python trace_stats.py --model-name <model_name>
python e2e.py --model-name <model_name>
```
