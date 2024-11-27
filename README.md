A performance model for predicting the training time of ML (DLRM, CV, NLP, etc) models on single-GPU and multi-GPU platforms. This is also the code repo for the following two papers:

(**TPDS 2024**) *Towards Universal Performance Modeling for Machine Learning Training on Multi-GPU Platforms. Zhongyi Lin, Ning Sun, Pallab Bhattacharya, Xizhou Feng, Louis Feng, John D. Owens.*

Bib:
```bibtex
@article{Lin:2024:TUP,
  author = {Zhongyi Lin and Ning Sun and Pallab Bhattacharya and
                  Xizhou Feng and Louis Feng and John D. Owens},
  title = {Towards Universal Performance Modeling for Machine
                  Learning Training on Multi-{GPU} Platforms},
  journal = {Transactions on Parallel and Distributed Systems},
  year = 2024,
  month = nov,
  code = {https://github.com/owensgroup/ml_perf_model},
  publisher = {IEEE},
  doi = {10.1109/TPDS.2024.3507814},
  url = {http://escholarship.org/uc/item/5mv1s1gg}
}
```

(**HiPC 2022**) *Building a Performance Model for Deep Learning Recommendation Model Training on GPUs. Zhongyi Lin, Louis Feng, Ehsan K. Ardestani, Jaewon Lee, John Lundell, Changkyu Kim, Arun Kejariwal, and John D. Owens.*

Bib:
```bibtex
@inproceedings{Lin:2022:BAP,
  author = {Zhongyi Lin and Louis Feng and Ehsan K. Ardestani
                  and Jaewon Lee and John Lundell and Changkyu Kim and
                  Arun Kejariwal and John D. Owens},
  title = {Building a Performance Model for Deep Learning
                  Recommendation Model Training on {GPU}s},
  booktitle = {2022 IEEE 29th International Conference on High
                  Performance Computing, Data, and Analytics},
  series = {HiPC 2022},
  year = 2022,
  month = dec,
  pages = {48--58},
  doi = {10.1109/hipc56025.2022.00019},
  url = {https://escholarship.org/uc/item/6rt535s6},
  publisher = {IEEE},
  eprint_ = {2201.07821v1},
  acceptance = {35/131 (26.7\%)},
  ucdcite = {a148}
}
```

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
