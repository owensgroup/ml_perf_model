# ml_perf_model
A continuation of Zhongyi's ML performance model work at Facebook.

To get started, simply run
```bash
git clone --recursive https://github.com/owensgroup/ml_perf_model.git
cd ml_perf_model/bench_params
./generate_benchmark_parameters.sh # Generate benchmark parameters (please modify the GPU memory size in the script).
cd ../sparse-ads-baselines
python setup.py install # Install table batched embedding lookup kernel.
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
```

Notice: This code also depends on the private `facebookexternal/ml_perf_model` repo.