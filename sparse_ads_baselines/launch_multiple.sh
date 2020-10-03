sbatch --error=result.8gpu.err --output result.8gpu.out --nodes=1 -p dev perf_test_multigpu.sh
sbatch --error=result.16gpu.err --output result.16gpu.out --nodes=2 -p priority perf_test_multigpu.sh
sbatch --error=result.24gpu.err --output result.24gpu.out --nodes=3 -p priority --comment pyper perf_test_multigpu.sh
sbatch --error=result.32gpu.err --output result.32gpu.out --nodes=4 -p priority --comment pyper perf_test_multigpu.sh
sbatch --error=result.48gpu.err --output result.48gpu.out --nodes=6 -p priority --comment pyper perf_test_multigpu.sh
sbatch --error=result.96gpu.err --output result.96gpu.out --nodes=12 -p priority --comment pyper perf_test_multigpu.sh
