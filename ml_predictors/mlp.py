import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import pandas as pd

class MLP(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(MLP, self).__init__()
        if not isinstance(n_hidden, list):
            n_hidden = list(n_hidden)
        prv = n_feature
        self.layers = torch.nn.ModuleList()
        for nxt in n_hidden:
            self.layers.append(torch.nn.Linear(prv, nxt))
            self.layers.append(torch.nn.Sigmoid())
            prv = nxt
        self.layers.append(torch.nn.Linear(prv, n_output))
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

# Utility functions
def abs_err(pred, real):
    return abs((pred - real) / real)

def err(pred, real):
    return (pred - real) / real

def gmae(x):
    return np.exp(np.log(abs(x)).mean())

def histogram(df, perc=True, is_abs=False, bins=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0]):
    count = len(df)
    ret = {}
    if is_abs:
        tmp_bins = []
        for i in range(0, len(bins) - 1):
            tmp_bins.append(-bins[len(bins) - 1 - i])
        for b in bins:
            tmp_bins.append(b)
        bins = tmp_bins
    for idx, b in enumerate(bins):
        if idx == 0:
            continue
        ret[(bins[idx-1], bins[idx])] = 0
    for x in df:
        for idx, b in enumerate(bins):
            if idx == 0:
                continue
            if x >= bins[idx-1] and x < bins[idx]:
                ret[(bins[idx-1], bins[idx])] += 1
                break
    for b, c in sorted(ret.items(), key=lambda x: x[0]):
        if perc:
            print("{:.0f}% - {:.0f}%: {:.2f}%".format(b[0] * 100, b[1] * 100, c / count * 100))
        else:
            print("{:.2f} - {:.2f}: {:.2f}%".format(b[0], b[1], c / count * 100))
    return ret

def strip_unit(x):
    for col in ['dram_read_throughput', 'dram_write_throughput', 'gld_requested_throughput', 'gld_throughput',\
               'gst_requested_throughput', 'gst_throughput', 'l2_read_throughput', 'l2_write_throughput', \
                'shared_load_throughput', 'shared_store_throughput']:
        if col in x.keys():
            if x[col].endswith('GB/s'):
                x[col] = float(x[col].rstrip('GB/s'))
            elif x[col].endswith('MB/s'):
                x[col] = float(x[col].rstrip('MB/s')) / 1e3
            elif x[col].endswith('B/s'):
                x[col] = float(x[col].rstrip('B/s')) / 1e9
            else:
                raise Exception("Unrecognizable unit!")
    return x
    
def p2f(x):
    for col in ['flop_dp_efficiency', 'flop_sp_efficiency', 'gld_efficiency', 'gst_efficiency', \
                'shared_efficiency', 'sm_efficiency', 'warp_execution_efficiency']:
        if col in x.keys():
            x[col] = float(str(x[col]).rstrip('%')) / 100.0
    return x

def strip_parenthesis(x):
    for col in ['dram_utilization', 'l2_utilization', 'tex_utilization']:
        if col in x.keys():
            x[col] = x[col].strip('(').strip(')')
    return x

def process_smem(x):
    # To bytes
    if 'smem' in x.keys():
        if x['smem'].endswith('MB'):
            x['smem'] = int(float(x['smem'].rstrip('MB')) * 1024 * 1024)
        elif x['smem'].endswith('KB'):
            x['smem'] = int(float(x['smem'].rstrip('KB')) * 1024)
        elif x['smem'].endswith('B'):
            x['smem'] = int(x['smem'].rstrip('B'))
        else:
            raise Exception("Unrecognizable unit!")
    return x
        
def preprocessing(df):
    df = df.apply(func=p2f, axis=1)
    df = df.apply(func=strip_unit, axis=1)
    df = df.apply(func=strip_parenthesis, axis=1)
    df = df.apply(func=process_smem, axis=1)
    df = df[(df['kernel_name'] != 'gemv2T_kernel') & (df['kernel_name'] != 'splitKreduce_kernel')]
    return df

def div_round_up(x, y):
    return int((x + y - 1) / y)

def choose(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# def gmae_loss(output, target):
#     x = abs((torch.exp(output) - torch.exp(target)) / torch.exp(target))
#     loss = torch.exp(torch.mean(torch.log(x)))
#     return loss

# def abs_mean_loss(output, target):
#     x = abs((output - target) / target)
#     loss = torch.mean(x)
#     return loss

BATCH_SIZE = 32 # 16
EPOCH = 800
peak_DRAM_BW = 804.497

def get_reshape_data():
    reshape_data = pd.read_csv('../data/reshape_1.csv', delimiter=',')
    reshape_data = preprocessing(reshape_data)
    r0 = reshape_data[reshape_data['trans_type'] == 0]
    r0 = r0[2 * (r0["batch_size"] * r0["M"] * r0["N"]) * 4 / r0['kernel_runtime'] / 1e3 <= peak_DRAM_BW]

    input_df = pd.DataFrame({
        'batch_size': np.log(r0['batch_size']),
        'M': np.log(r0['M']),
        'N': np.log(r0['N'])
    })
    x = torch.tensor(input_df.values).float()

    output_df = pd.DataFrame({
        'kernel_runtime': np.log(r0['kernel_runtime'])
    })
    y = torch.tensor(output_df.values).float()

    return 3, x, y

def get_fc_data():
    fc_data = pd.read_csv('../data/fully_connected_1.csv', delimiter=',')
    f0 = preprocessing(fc_data)
    f0 = f0[2 * f0["batch_size"] * (f0["M"] * f0["N"] + f0["M"] * f0["K"] + f0["N"] * f0["K"]) * 4 / f0['kernel_runtime'] / 1e3 <= peak_DRAM_BW]

    input_df = pd.DataFrame({
        'batch_size': np.log(f0['batch_size']),
        'M': np.log(f0['M']),
        'N': np.log(f0['N']),
        'K': np.log(f0['K'])
    })
    x = torch.tensor(input_df.values).float()

    output_df = pd.DataFrame({
        'kernel_runtime': np.log(f0['kernel_runtime'])
    })
    y = torch.tensor(output_df.values).float()

    return 4, x, y

##################################
# op_tpye = "reshape"
op_type = "fc"
##################################

if op_type == "reshape":
    n_feature, x, y = get_reshape_data()
else:
    n_feature, x, y = get_fc_data()

print(x.size())
print(y.size())

op_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=op_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0)

min_loss = 1e9
best_config = None
for size in [128, 256, 512, 1024]:
    for num_layers in [3, 4, 5, 6, 7]:
        for lr in [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]:
            for opt in ['adam', 'sgd']:
                for loss_func in [torch.nn.MSELoss()]:
                    learning_rate = lr * 10 if opt == 'sgd' else lr
                    print("Size: {}, num_layers: {}, learning rate: {}, optimizer: {}, loss function: {}".format(size, num_layers, learning_rate, opt, loss_func))

                    size_tuple = tuple([size] * num_layers)
                    net = MLP(n_feature=n_feature, n_hidden=size_tuple, n_output=1)
                    net.apply(init_weights)
                    if opt == 'adam':
                        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, eps=1e-8, weight_decay=1e-4, amsgrad=False)
                    else: # SGD
                        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

                    for epoch in range(EPOCH):
                        for step, (batch_x, batch_y) in enumerate(loader):
                            b_x = Variable(batch_x)
                            b_y = Variable(batch_y)

                            prediction = net(b_x)
                            loss = loss_func(prediction, b_y)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                        if epoch % 50 == 0:
                            print("******* Epoch {} *******".format(epoch))
                            prediction = net(x)
                            loss = loss_func(prediction, y)
                            print("Loss: {}".format(loss))

                    estimated_time = torch.exp(net(x).detach().view(-1))
                    error = abs_err(estimated_time, torch.exp(y).view(-1))
                    histogram(error, is_abs=True)
                    print("GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(gmae(error) * 100.0, error.mean() * 100.0, error.std() * 100.0))
                    if gmae(error) < min_loss:
                        min_loss = gmae(error)
                        best_config = (size, num_layers, learning_rate, opt, loss_func)
                        print("Current best config to {}".format(best_config))
                        torch.save(net.state_dict(), "./predictor_{}.pth".format(op_type))

print("Min gmae loss: {}".format(min_loss))
print("Best config: {}".format(best_config))
