# ml_perf_model
A continuation of Zhongyi's ML performance model work at Facebook.

To get started, simply run
```
git clone --recursive https://github.com/owensgroup/ml_perf_model.git
cd ml_perf_model/bench_params
./generate_benchmark_parameters.sh
./benchmark.sh fully_connected 1 # 1 for forward
```