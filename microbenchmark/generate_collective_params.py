# BSD 3-Clause License
#
# Copyright (c) 2021, The Regents of the University of California, Davis
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import subprocess, os
from analysis.memory_bw_utils import *
from analysis.utils import GPU_COUNT, GPU_NAME

superscript = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
collectives = ['all_to_allv', 'all_reduce']
data = process_param_data(
    prefix="../3rdparty/param/train/comms/pt/bench_results",
    collectives=collectives,
    num_gpus=GPU_COUNT,
)

# Get slopes of for the 2nd section of the curves
mem_chs = {}
sigmoid_params = {}
for collective in collectives:
    mem_chs[collective] = get_memory_characteristics(data[collective])
    sigmoid_params[collective] = fit_sigmoid_bw_predictor(data[collective], mem_chs[collective])
topology = subprocess.Popen(["nvidia-smi", "topo", "-m"], stdout=subprocess.PIPE).stdout.read()
topology = str(topology).replace('\\x1b[4m', '').replace('\\x1b[0m', '').replace('\\t', '\\t\\t')
topology = bytes(topology, "utf-8").decode("unicode_escape")[2:-1]


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

mem_param_file = "../analysis/mem_comm_params/{}x{}.py".format(GPU_COUNT, GPU_NAME)
if not os.path.exists(mem_param_file):
    with open(mem_param_file, 'w') as f:
        f.write(template)
