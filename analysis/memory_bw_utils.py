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

    # Mark the saturation point with 95% of peak
    for idx in reversed(range(num_samples)):
        if idx == 0 or idx == num_samples - 1:
            continue
        if data[bw][idx] < (1 - ratio_th) * max_bw:
            saturation_idx = idx
            break

    # Mark the first jump of latency as increment_idx
    increment_idx = 4
    for idx in range(num_samples):
        if idx < 5 or idx == num_samples - 1:
            continue
        if abs(latency_ratios[idx] - 1) > ratio_th * 2:
            increment_idx = idx - 1
            break

    return increment_idx, saturation_idx


def get_memory_characteristics(collective_data, size='size', bw='bus_bw', latency='latency'):
    ln_idx, sats_idx = get_turning_points(collective_data, bw=bw, latency=latency)
    ln_p, sats_p = int(np.log2(collective_data[size][ln_idx])), int(np.log2(collective_data[size][sats_idx])) # log sizes of both ln point and sats point
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


def fit_sigmoid_bw_predictor(collective_data, size='size', bw='bus_bw'):
    xdata = np.log2(collective_data[size])
    ydata = np.log10(collective_data[bw])
    p0 = [
        ydata.max(), 
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


def predict_bw(size, mul_factor, mem_ch, sigmoid_param):
    log_size = np.log2(size)
    if log_size <= mem_ch["ln_p"]:
        return size / mem_ch["overhead"] / 1e3 # us -> GB/s
    elif log_size >= mem_ch["sats_p"]:
        return mem_ch["max_bw"]
    else:
        bw = get_sigmoid_bw(log_size, sigmoid_param)
        return bw / mul_factor


# In us
def predict_data_movement_time(size, mul_factor, mem_ch, sigmoid_param):
    log_size = np.log2(size)
    if log_size <= mem_ch["ln_p"]:
        return mem_ch["overhead"]
    elif log_size >= mem_ch["sats_p"]:
        return size / mem_ch["max_bw"] * mul_factor / 1e3 + mem_ch["overhead"] # GB/s -> us
    else:
        bw = get_sigmoid_bw(log_size, sigmoid_param)
        return size / bw * mul_factor / 1e3 # GB/s -> us
