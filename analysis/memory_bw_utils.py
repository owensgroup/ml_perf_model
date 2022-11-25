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

import numpy as np
from scipy.optimize import curve_fit

# alg_bw -> bus_bw for multi-gpu collectives
MUL_FACTOR_FUNCS = {
    'all_to_all': lambda n: (n-1) / n,
    'all_to_allv': lambda n: (n-1) / n,
    'all_reduce': lambda n: 2 * (n-1) / n,
    'all_gather': lambda n: (n-1) / n,
    'all_gather_base': lambda n: (n-1) / n,
    'reduce': lambda n: 1,
    'reduce_scatter': lambda n: (n-1) / n,
    'others': lambda n: 1,
}


# Get turning points of the bus BW curve for a collective
def get_turning_points(data, ratio_th=0.05, bw='bus_bw', latency='latency'):
    num_samples = len(data)
    max_bw = data[bw].max()
    bw_ratios = []
    latency_ratios = []
    for idx in range(num_samples):
        if idx == 0 or idx == num_samples - 1:
            bw_ratios.append(-1)
            latency_ratios.append(-1)
            continue
        bw_ratios.append(data[bw][idx-1] * data[bw][idx+1] / data[bw][idx] / data[bw][idx])
        latency_ratios.append(data[latency][idx-1] * data[latency][idx+1] / data[latency][idx] / data[latency][idx])

    # Mark the saturation point as two dps after 95% of the peak
    saturation_idx = num_samples - 1
    for idx in reversed(range(num_samples)):
        if idx == 0 or idx == num_samples - 1:
            continue
        if data[bw][idx] < (1 - ratio_th / 2) * max_bw:
            saturation_idx = min(idx + 2, num_samples - 1)
            break

    # Mark the first jump of latency as linearity_idx
    linearity_idx = 4
    for idx in range(num_samples):
        if idx < 5 or idx == num_samples - 1:
            continue
        if abs(latency_ratios[idx] - 1) > ratio_th * 2:
            linearity_idx = idx - 1
            break

    return linearity_idx, saturation_idx


def get_memory_characteristics(collective_data, size='size', bw='bus_bw', latency='latency'):
    ln_idx, sats_idx = get_turning_points(collective_data, bw=bw, latency=latency)
    ln_p, sats_p = np.log2(collective_data[size][ln_idx]), np.log2(collective_data[size][sats_idx]) # log sizes of both ln point and sats point
    max_bus_bw = collective_data[bw].max() # Pick the bw for the 3rd section of the curves
    overhead = collective_data[latency][max(ln_idx - 1, 0)]

    return {
        "ln_p": ln_p,
        "sats_p": sats_p,
        "max_bw": max_bus_bw,
        "overhead": overhead,
    }


def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return y


def fit_sigmoid_bw_predictor(collective_data, mem_ch, size='size', bw='bus_bw'):
    d = collective_data[
        (collective_data[size] >= 2 ** mem_ch["ln_p"]) &
        (collective_data[size] <= 2 ** mem_ch["sats_p"])
    ]
    xdata = np.log2(d[size])
    ydata = np.log10(d[bw])
    p0 = [
        np.log2(collective_data[size]).max(), 
        np.median(xdata),
        1,
        ydata.min()
    ] # this is an mandatory initial guess
    popt, _ = curve_fit(sigmoid, xdata, ydata, p0, method='dogbox')
    return popt


def get_linear_bw(s, s0, k, b):
    return 10 ** ((s - s0) * k) * b


def get_sigmoid_bw(s, sigmoid_param):
    return 10 ** sigmoid(s, *sigmoid_param)


# In GB/s
def predict_bw(size, mul_factor, mem_ch, sigmoid_param):
    log_size = np.log2(size)
    if log_size <= mem_ch["ln_p"]:
        return size / mem_ch["overhead"] / 1e3 # us -> GB/s
    elif log_size > mem_ch["sats_p"]:
        return mem_ch["max_bw"]
    else:
        bw = get_sigmoid_bw(log_size, sigmoid_param)
        return bw / mul_factor


# In us
def predict_data_movement_time(size, mul_factor, mem_ch, sigmoid_param):
    log_size = np.log2(size)
    if log_size <= mem_ch["ln_p"]:
        return mem_ch["overhead"]
    elif log_size > mem_ch["sats_p"]:
        return size / mem_ch["max_bw"] * mul_factor / 1e3 + mem_ch["overhead"] # GB/s -> us
    else:
        bw = get_sigmoid_bw(log_size, sigmoid_param)
        return size / bw * mul_factor / 1e3 # GB/s -> us
