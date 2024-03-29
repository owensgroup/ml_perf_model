A performance model for predicting the training time of ML (DLRM, CV, NLP, etc) models on single-GPU and multi-GPU platforms.

The execution graph observer has been integrated into Pytorch since [d76af8f41c](https://github.com/pytorch/pytorch/tree/d76af8f41c6404b090b13ab9a868a71423d6d298) (nightly build at 2022/05/31).

### Prerequisites and Installation
```bash
./prereq.sh
```

### Microbenchmark
```bash
cd ${PM_HOME}/microbenchmark/nsight # Or ${PM_HOME}/microbenchmark/nvprof, depending on the choice of profiler
./microbenchmark.sh fully_connected 1 # 1 for forward
./microbenchmark.sh fully_connected 1 1 # The second 1 for big batch size.
./microbenchmark.sh embedding_lookup 1
./microbenchmark.sh embedding_lookup 0
./microbenchmark.sh embedding_lookup 1 1 # The second 1 for big batch size.
./microbenchmark.sh embedding_lookup 0 1
./microbenchmark.sh embedding_lookup 1 0 2 # Benchmark with FBGEMM open-source dataset
./microbenchmark.sh embedding_lookup 0 0 2
./microbenchmark.sh conv2d 1 # We also support convolution and BN for comparison with other performance models on DL models other the DLRM.
./microbenchmark.sh conv2d 1 1
./microbenchmark.sh conv1d 1
./microbenchmark.sh conv1d 0
./microbenchmark.sh concat 1
./microbenchmark.sh memcpy 1
./microbenchmark.sh transpose 1
./microbenchmark.sh bn 1
./microbenchmark.sh bn 0
./microbenchmark.sh ln 1
./microbenchmark.sh ln 0
./microbenchmark.sh dropout 1
./microbenchmark.sh tril 1
./microbenchmark.sh tril 0
```

### Communication collective microbenchmark (with PARAM)
```bash
cd ${PM_HOME}/microbenchmark
./collective_bench.sh
cd ..
```

### Training ML-based kernel performance model
```bash
python mlp.py --op-type fully_connected --batch-size 64
python mlp.py --op-type embedding_lookup --batch-size 64 --epoch 1200
python mlp.py --op-type embedding_lookup --batch-size 64 --epoch 1200 --backward
python mlp.py --op-type conv2d --batch-size 16 --epoch 1200
python mlp.py --op-type conv2d --batch-size 16 --epoch 1200 --backward
python mlp.py --op-type conv1d --batch-size 32
python mlp.py --op-type conv1d --batch-size 32 --backward
python mlp.py --op-type transpose --batch-size 32
python mlp.py --op-type bn --batch-size 32
python mlp.py --op-type bn --batch-size 32 --backward
python mlp.py --op-type ln --batch-size 64
python mlp.py --op-type ln --batch-size 64 --backward
python mlp.py --op-type dropout --batch-size 64
python mlp.py --op-type tril --epoch 1000
python mlp.py --op-type tril --epoch 2000 --backward
```
To print all performance model error rates after training is done, run:
```bash
python kernel_pm_acc.py # Include all above + all_to_all and all_reduce
```

### Execution graph extraction and profiler trace generation
```bash
cd ${PM_HOME}/benchmark
./dlrm_benchmark.sh -m <model_name> -b <batch_size>
./convnet_benchmark.sh <model_name> <batch_size>
./nlp_benchmark.sh -m <model_name> -b <batch_size>
./rm_benchmark.sh <model_name> <batch_size>
```

### Trace analysis and E2E runtime prediction
```bash
cd ${PM_HOME}/benchmark
python trace_stats.py --model-name <model_name> --num-gpus <num> # Currently only single-GPU is supported
python e2e.py --model-name <model_name> --num-gpus <num> # Ditto
```

### One-liner for E2E benchmark, trace analysis and runtime prediction
```bash
# Run all above together, with DLRM_random, DLRM_MLPerf, DLRM_DDP, and DLRM_open_source (with a default selection of tables).
./run_experiments.sh -o -r
```

### One-liner for MULTI_GPU E2E benchmark, trace analysis and runtime prediction
```bash
# Run all above together on multi-GPU platforms, with DLRM_open_source and randomly generated embedding tables.
./run_random_experiments.sh -o -r
```
