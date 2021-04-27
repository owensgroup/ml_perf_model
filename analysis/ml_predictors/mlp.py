import torch
from torch.autograd import Variable, backward
import torch.utils.data as Data
import numpy as np
import pandas as pd
import argparse, json, os

pm_home = os.environ.get('PM_HOME')
if pm_home is None:
    pm_home = "/home/m092926/daisy/Documents/ml_perf_model"
dir_prefix = "{}/analysis/ml_predictors".format(pm_home)


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


def dash_separated_ints(value):
    vals = value.split("-")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value
            )
    return value


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


def get_data(op_type, backward=False):
    data = pd.read_csv('{}/../../data/{}_{}.csv'.format(dir_prefix, op_type, 1 if not backward else 0), delimiter=',')
    data = preprocessing(data)

    if op_type == 'fully_connected':
        data = data[data['kernel_name'].str.contains("sgemm")] # Train on samples with 'sgemm' in kernel name
        input_df = pd.DataFrame({
            'batch_size': np.log(data['batch_size']),
            'M': np.log(data['M']),
            'N': np.log(data['N']),
            'K': np.log(data['K'])
        })
    elif op_type == 'transpose':
        input_df = pd.DataFrame({
            'batch_size': np.log(data['batch_size']),
            'M': np.log(data['M']),
            'N': np.log(data['N'])
        })
    else: # tril
        if backward:
            data = data[['batch_size', 'M', 'N', 'diag', 'kernel_runtime']].groupby(['batch_size', 'M', 'N', 'diag'], as_index=False).sum() # Sum up all kernels
        input_df = pd.DataFrame({
            'batch_size': np.log(data['batch_size']),
            'M': np.log(data['M']),
            'N': np.log(data['N']),
            'diag': data['diag']
        })

    x = torch.tensor(input_df.values).float()

    output_df = pd.DataFrame({
        'kernel_runtime': np.log(data['kernel_runtime'])
    })
    y = torch.tensor(output_df.values).float()

    return x.shape[1], x.to('cuda:0'), y.to('cuda:0')


def get_pretrained_net(op_type, backward=False):
    with open("{}/best_config_{}_{}.json".format(dir_prefix, op_type, 1 if not backward else 0), "r") as f:
        best_config = json.load(f)
        n_hidden = [best_config["size"]] * best_config["num_layers"]
    if op_type == "fully_connected":
        n_feature = 4
    elif op_type == "transpose":
        n_feature = 3
    else: # tril
        n_feature = 4
    net = MLP(n_feature=n_feature, n_hidden=n_hidden, n_output=1)
    net.load_state_dict(torch.load("{}/predictor_{}_{}.pth".format(dir_prefix, op_type, 1 if not backward else 0)))
    return net


def inference(op_type, input_sizes, backward=False):
    net = get_pretrained_net(op_type, backward)

    input_sizes = [int(x) for x in input_sizes.split('-')]
    if op_type == "fully_connected":
        n_feature = 4
        input_sizes = [np.log(x) for x in input_sizes]
    elif op_type == "transpose":
        n_feature = 3
        input_sizes = [np.log(x) for x in input_sizes]
    else: # tril
        n_feature = 4
        input_sizes = [np.log(x) for x in input_sizes[:-1]] + [input_sizes[-1]]
    assert len(input_sizes) == n_feature
    
    return torch.exp(net(torch.tensor(input_sizes).float()).cpu().detach().view(-1)).item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training MLP performance model for FC, transpose, and tril.")
    parser.add_argument("--op-type", type=str, required=True)
    parser.add_argument("--backward", action="store_true", default=False) # For tril
    parser.add_argument("--inference", action="store_true", default=False)
    parser.add_argument("--inference-input-sizes", type=dash_separated_ints, default="1-2-3")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=800)
    args = parser.parse_args()

    assert args.op_type in ["fully_connected", "transpose", "tril"]
    suffix = "{}_{}".format(args.op_type, 1 if not args.backward else 0)

    if args.inference:
        t = inference(args.op_type, args.inference_input_sizes, args.backward)
        print("Predicted inference time of {} with input sizes {}: {:.2f}.".format(suffix, args.inference_input_sizes, t))
        exit()

    n_feature, x, y = get_data(op_type=args.op_type)
    op_dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset=op_dataset,
        batch_size=args.batch_size,
        shuffle=True, num_workers=0)
    print("Op type: {}, dataset length: {}, batch size: {}, epoch: {}".format(args.op_type, y.shape[0], args.batch_size, args.epoch))

    if os.path.exists("{}/best_config_{}.json".format(dir_prefix, suffix)):
        with open("{}/best_config_{}.json".format(dir_prefix, suffix), "r") as f:
            best_config = json.load(f)
        net = get_pretrained_net(args.op_type, args.backward)
        estimated_time = torch.exp(net(x.cpu()).detach().view(-1))
        error = abs_err(estimated_time, torch.exp(y.cpu().detach()).view(-1))
        min_error = gmae(error)
        print("Pretrained net error: GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(gmae(error) * 100.0, error.mean() * 100.0, error.std() * 100.0))
    else:
        best_config = None
        min_error = 1e9
    for size in [128, 256, 512, 1024]:
        for num_layers in [3, 4, 5, 6, 7]:
            for lr in [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]:
                for opt in ['adam', 'sgd']:
                    for loss_func in [torch.nn.MSELoss()]:
                        learning_rate = lr * 10 if opt == 'sgd' else lr
                        print("Size: {}, num_layers: {}, learning rate: {}, optimizer: {}, loss function: {}".format(size, num_layers, learning_rate, opt, loss_func))

                        size_tuple = tuple([size] * num_layers)
                        net = MLP(n_feature=n_feature, n_hidden=size_tuple, n_output=1).to('cuda:0')
                        net.apply(init_weights)
                        if opt == 'adam':
                            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, eps=1e-8, weight_decay=1e-4, amsgrad=False)
                        else: # SGD
                            optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

                        for epoch in range(args.epoch):
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

                        estimated_time = torch.exp(net(x).cpu().detach().view(-1))
                        error = abs_err(estimated_time, torch.exp(y.cpu().detach()).view(-1))
                        histogram(error, is_abs=True)
                        print("GMAE: {:.2f}%, mean: {:.2f}%, std: {:.2f}%".format(gmae(error) * 100.0, error.mean() * 100.0, error.std() * 100.0))
                        if gmae(error) == 0.0:
                            print("Something wrong here! Not saving anything.")
                            print(abs(error))
                            continue
                        if gmae(error) < min_error:
                            min_error = gmae(error)
                            best_config = {
                                "size": size,
                                "num_layers": num_layers,
                                "lr": learning_rate,
                                "optimizer": opt,
                                "loss_fn": loss_func.__class__.__name__
                            }
                            torch.save(net.state_dict(), "{}/predictor_{}.pth".format(dir_prefix, suffix))
                            torch.save(optimizer.state_dict(), "{}/optimizer_{}.pth".format(dir_prefix, suffix))
                            with open('{}/best_config_{}.json'.format(dir_prefix, suffix), 'w') as f:
                                json.dump(best_config, f)
                        with open('{}/errors_{}.csv'.format(dir_prefix, suffix), 'a') as f:
                            if not os.path.exists('{}/errors_{}.csv'.format(dir_prefix, suffix)):
                                f.write("size,num_layers,lr,optimizer,loss_fn,GMAE,mean,std\n")
                            f.write("{},{},{},{},{},{:.4f},{:.4f},{:.4f}\n".format(size, num_layers, lr, opt, loss_func.__class__.__name__, gmae(error), error.mean(), error.std()))

                        print("Current best config is {}, with error {:.2f}%".format(best_config, min_error * 100.0))

    print("Min gmae loss: {}".format(min_error))
    print("Best config: {}".format(best_config))
