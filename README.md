# ml_perf_model
A continuation of Zhongyi's ML performance model work at Facebook.

To get started, simply run
```
git clone --recursive https://github.com/owensgroup/ml_perf_model.git
cd ml_perf_model/bench_params
./generate_benchmark_parameters.sh # Generate benchmark parameters (please modify the GPU memory size in the script).
cd ../sparse-ads-baselines
python setup.py install # Install table batched embedding lookup kernel.
cd ..
source ./init_vars.sh # Turn off turbo, turn on performance, lock frequency, etc.
./benchmark.sh fully_connected 1 # 1 for forward
# ./benchmark.sh fully_connected 1 1 # The second 1 for big batch size.
# ./benchmark.sh reshape 1
# ./benchmark.sh embedding_lookup 1
# ./benchmark.sh embedding_lookup 0
```
