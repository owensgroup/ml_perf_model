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
def get_turning_points(bw_data, ratio_th=0.05):
    num_samples = len(bw_data)
    max_bw = bw_data.max()
    ratios = []
    for idx in range(num_samples):
        if idx == 0 or idx == num_samples - 1:
            ratios.append(-1)
            continue
        bw = bw_data[idx]
        prev_bw = bw_data[idx-1]
        next_bw = bw_data[idx+1]
        ratios.append(next_bw * prev_bw / bw / bw)

    # Mark the saturation point with 95% of peak
    for idx in reversed(range(num_samples)):
        if idx == 0 or idx == num_samples - 1:
            continue
        if bw_data[idx] < (1 - ratio_th) * max_bw:
            saturation_idx = idx
            break

    # Mark the increment point with 3 consecutive flat samples
    count = 0
    for idx in range(num_samples):
        if idx == 0 or idx == num_samples - 1:
            continue
        if abs(ratios[idx] - 1) <= ratio_th:
            count += 1
            if count >= 3:
                increment_idx = idx
                break
        else:
            count = 0

    return increment_idx, saturation_idx


def get_feature(collective_data, size='size', bw='bus_bw', latency='latency', all_features=False):
    incr_idx, sats_idx = get_turning_points(collective_data[bw])
    incr_p, sats_p = int(np.log2(collective_data[size][incr_idx])), int(np.log2(collective_data[size][sats_idx])) # log sizes of both incr point and sats point
    max_bus_bw = collective_data[bw].max() # Pick the bw for the 3rd section of the curves
    min_bus_bw = collective_data[bw][incr_idx] # Pick the bw for the 1st section of the curves
    slope = (np.log10(max_bus_bw / min_bus_bw)) / (sats_p - incr_p)
    overhead = collective_data[latency][0]
    if all_features:
        return (incr_p, sats_p, slope, min_bus_bw, max_bus_bw, overhead)
    return (incr_p, sats_p, max_bus_bw, overhead)


def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return y


def fit_sigmoid_bw_predictor(collective_data, size='size', bw='bus_bw'):
    xdata = np.log2(collective_data[size])
    ydata = np.log10(collective_data[bw])
    p0 = [ydata.max(), 
            np.median(xdata),
            1,
            ydata.min()] # this is an mandatory initial guess
    popt, _ = curve_fit(sigmoid, xdata, ydata, p0, method='dogbox')
    return popt


def get_linear_bw(s, s0, k, b):
    return 10 ** ((s - s0) * k) * b


def get_sigmoid_bw(s, sigmoid_param):
    return 10 ** sigmoid(s, *sigmoid_param)


# def predict_linear(size, mul_factor, incr_p, sats_p, slope, min_bw, max_bw, overhead):
#     log_size = np.log2(size)
#     if log_size <= incr_p:
#         return overhead
#     elif log_size >= sats_p:
#         return size / max_bw * mul_factor / 1e3 + overhead
#     else:
#         bw = get_linear_bw(log_size, incr_p, slope, min_bw)
#         return size / bw * mul_factor / 1e3 + overhead


# Sigmoid
def predict_data_movement_time(size, mul_factor, incr_p, sats_p, max_bw, overhead, sigmoid_param):
    log_size = np.log2(size)
    if log_size <= incr_p:
        return overhead
    elif log_size >= sats_p:
        return size / max_bw * mul_factor / 1e3 + overhead
    else:
        bw = get_sigmoid_bw(log_size, sigmoid_param)
        return size / bw * mul_factor / 1e3
