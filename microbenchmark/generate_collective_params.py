import re, subprocess, os
import numpy as np
from analysis.memory_bw_utils import *
from analysis.utils import GPU_COUNT, GPU_NAME

num_gpus = 4
epsilon = 4e-4
superscript = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
collectives = ['all_to_allv', 'all_reduce']

num_of_collectives = len(collectives)
data = {}
for idx, collective in enumerate(collectives):
    data[collective] = {
        'size': [],
        'latency': [],
        'alg_bw': [],
        'bus_bw': []
    }
    filename = '/home/m092926/daisy/Documents/ml_perf_model/3rdparty/param/train/comms/pt/bench_results/{}_{}.txt'.format(collective, num_gpus)
    header_found = False
    with open(filename, 'r') as f:
        print("Processing {} data...".format(collective))
        lines = f.readlines()
        for line in lines:
            if re.search('COMMS-RES', line):
                if not header_found:
                    header_found = True
                    continue
                data[collective]['size'].append(int(line.split('\t')[2].lstrip('\t')))
                data[collective]['latency'].append(float(line.split('\t')[4].lstrip('\t')))
                data[collective]['alg_bw'].append(float(line.split('\t')[-2].rstrip('\n')) + epsilon)
                data[collective]['bus_bw'].append(float(line.split('\t')[-1].rstrip('\n')) + epsilon)
        data[collective]['size'] = np.array(data[collective]['size'])
        data[collective]['latency'] = np.array(data[collective]['latency'])
        data[collective]['alg_bw'] = np.array(data[collective]['alg_bw'])
        data[collective]['bus_bw'] = np.array(data[collective]['bus_bw'])

# Get slopes of for the 2nd section of the curves
mem_chs = {}
sigmoid_params = {}
for idx, collective in enumerate(collectives):
    mem_chs[collective] = get_memory_characteristics(data[collective])
    sigmoid_params[collective] = fit_sigmoid_bw_predictor(data[collective])
topology = subprocess.Popen(["nvidia-smi", "topo", "-m"], stdout=subprocess.PIPE).stdout.read()

template = f'''from analysis.memory_bw_utils import MUL_FACTOR_FUNCS
"""
Performance modeling parameters for the following communication topology and config:

GPUs: {GPU_COUNT} x {GPU_NAME}
Topology:
{topology}
"""

ALL_TO_ALL_PARAMS = {{
    "mul_factor": MUL_FACTOR_FUNCS["all_to_allv"]({GPU_COUNT}),
    "mem_ch": {{
        'ln_p': { mem_chs["all_to_allv"]["ln_p"] },
        'sats_p': { mem_chs["all_to_allv"]["sats_p"] },
        'max_bw': { mem_chs["all_to_allv"]["max_bw"] },
        'overhead': { mem_chs["all_to_allv"]["overhead"] },
    }},
    "sigmoid_param": (
        { sigmoid_params["all_to_allv"][0] },
        { sigmoid_params["all_to_allv"][1] },
        { sigmoid_params["all_to_allv"][2] },
        { sigmoid_params["all_to_allv"][3] },
    ),
}}

ALL_REDUCE_PARAMS = {{
    "mul_factor": MUL_FACTOR_FUNCS["all_reduce"]({GPU_COUNT}),
    "mem_ch": {{
        'ln_p': { mem_chs["all_reduce"]["ln_p"] },
        'sats_p': { mem_chs["all_reduce"]["sats_p"] },
        'max_bw': { mem_chs["all_reduce"]["max_bw"] },
        'overhead': { mem_chs["all_reduce"]["overhead"] },
    }},
    "sigmoid_param": (
        { sigmoid_params["all_reduce"][0] },
        { sigmoid_params["all_reduce"][1] },
        { sigmoid_params["all_reduce"][2] },
        { sigmoid_params["all_reduce"][3] },
    ),
}}

'''

param_file = "../analysis/{}x{}.py".format(GPU_COUNT, GPU_NAME)
if not os.path.exists(param_file):
    with open(param_file, 'w') as f:
        f.write(template)
