# dlrm_gpu_perf_model
DLRM GPU Training Performance Model

### Prerequisites and Installation
```bash
pip install h5py tqdm sklearn tensorboard torchviz onnx gputil spacy dill # Dependencies
git clone --recursive https://github.com/owensgroup/ml_perf_model.git

cd ml_perf_model/3rdparty/bench_params
./generate_benchmark_parameters.sh # Generate benchmark parameters (please modify the GPU memory size in the script).
cd ../sparse-ads-baselines
python setup.py install # Install table batched embedding lookup kernel.
cd ../mlperf-logging
python setup.py install # Install MLPerf logging for DLRM
cd ../param/train/comms/pt
./init.sh # Initialization for PARAM
cd ../../../../../
source ./init_vars.sh # Turn off turbo, turn on performance, lock frequency, etc.

# Torchvision for ConvNet benchmark
cd ..
git clone https://github.com/pytorch/vision.git torchvision
cd torchvision
python setup.py clean --all install
cd ../ml_perf_model

# Torchtext for NLP benchmark
cd ..
git clone https://github.com/pytorch/text.git torchtext
cd torchtext
git submodule update --init --recursive
python setup.py clean --all install
cd ../ml_perf_model
```

### Microbenchmark
```bash
cd ${PM_HOME}/microbenchmark/nvprof # Or ${PM_HOME}/microbenchmark/nsight, depending on the choice of profiler
./microbenchmark.sh fully_connected 1 # 1 for forward
./microbenchmark.sh fully_connected 1 1 # The second 1 for big batch size.
./microbenchmark.sh concat 1
./microbenchmark.sh memcpy 1
./microbenchmark.sh transpose 1
./microbenchmark.sh embedding_lookup 1
./microbenchmark.sh embedding_lookup 0
./microbenchmark.sh embedding_lookup 1 1 # The second 1 for big batch size.
./microbenchmark.sh embedding_lookup 0 1
./microbenchmark.sh tril 1
./microbenchmark.sh tril 0
./microbenchmark.sh conv 1 # We also support convolution and BN for comparison with other performance models on DL models other the DLRM.
./microbenchmark.sh conv 1 1
./microbenchmark.sh bn 1
./microbenchmark.sh bn 0
```

### Training ML-based kernel performance model
```bash
python mlp.py --op-type fully_connected --batch-size 64
python mlp.py --op-type conv --batch-size 16 --epoch 1200
python mlp.py --op-type conv --backward --batch-size 16 --epoch 1200
python mlp.py --op-type transpose --batch-size 32
python mlp.py --op-type bn --batch-size 32
python mlp.py --op-type bn --backward --batch-size 32
python mlp.py --op-type tril --epoch 1000
python mlp.py --op-type tril --backward --epoch 2000
```
To print all performance model error rates after training is done, run:
```bash
python kernel_pm_acc.py
```

### Execution graph extraction and profiler trace generation
```bash
cd ${PM_HOME}/benchmark
./dlrm_benchmark.sh <model_name> <batch_size>
./convnet_benchmark.sh <model_name> <batch_size>
./nlp_benchmark.sh <model_name> <batch_size>
```
Notice: This code also depends on the private `facebookexternal/ml_perf_model` repo.

### Trace analysis and E2E runtime prediction
```bash
cd ${PM_HOME}/benchmark
python trace_stats.py --model-name <model_name> --num-gpus <num> # Currently only single-GPU is supported
python e2e.py --model-name <model_name> --num-gpus <num> # Ditto
```

### One-liner for E2E benchmark, trace analysis and runtime prediction
```bash
# Run all above together
./run_experiments.sh
```
