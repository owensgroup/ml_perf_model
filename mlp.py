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

import torch
import torch.utils.data as Data
from torch.autograd import Variable
import argparse, json, os
from analysis.utils import *
from analysis.inference import infer


class FBGEMMDataset(Data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataset(x, y, fbgemm=False):
    x = [xx.cuda() for xx in x] if isinstance(x, list) else x.cuda()
    y = [yy.cuda() for yy in y] if isinstance(y, list) else y.cuda()
    if fbgemm:
        return FBGEMMDataset(x, y)
    return Data.TensorDataset(x, y)


# TODO: All backward models should be trained with an extra version of weight-only for topologically the first ops in a model
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training MLP performance model for FC, conv2d, conv1d, transpose, BN, and tril.")
    parser.add_argument("--op-type", type=str, required=True)
    parser.add_argument("--backward", action="store_true", default=False) # For conv2d/conv1d/bn/tril
    parser.add_argument("--inference", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=800)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--emb-data-path-suffix", type=str, default='fbgemm_dlrm_datasets')
    parser.add_argument("--emb-table-configs-path", type=str, default='/nvme/deep-learning/dlrm_datasets/embedding_bag/2021/fbgemm_t856_bs65536_configs.json')
    args = parser.parse_args()

    assert args.op_type in [
        "embedding_lookup",
        "fully_connected",
        "conv2d",
        "conv1d",
        "transpose",
        "bn",
        "tril"
    ]
    is_emb = args.op_type=="embedding_lookup"
    suffix = "{}_{}".format(args.op_type, 1 if not args.backward else 0)
    n_feature, train_x, train_y, test_x, test_y = get_train_test_data(
        op_type=args.op_type,
        backward=args.backward,
        test_frac=args.test_frac,
        suffix=args.emb_data_path_suffix,
        table_configs=args.emb_table_configs_path)

    train_dataset = get_dataset(train_x, train_y, fbgemm=is_emb)
    loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=(
            lambda x: ([xx[0] for xx in x], torch.stack([xx[1] for xx in x], dim=0))
        ) if is_emb else None,
        shuffle=True,
        num_workers=0,
    )
    print("Device: {}, op type: {}, train dataset length: {}, batch size: {}, epoch: {}".format(GPU_NAME, args.op_type, len(train_dataset), args.batch_size, args.epoch))

    suffix = "{}_{}".format(args.op_type, 1 if not args.backward else 0)
    if os.path.exists("{}/analysis/ml_predictors/{}/best_config_{}.json".format(PM_HOME, GPU_NAME, suffix)):
        best_config, min_error = infer(
            args.op_type,
            backward=args.backward,
            emb_use_mlp=True,
            suffix=args.emb_data_path_suffix,
            table_configs=args.emb_table_configs_path
        )
        if args.inference:
            exit()
    else:
        best_config = None
        min_error = 1e9
    for size in [64, 128, 256, 512]:
        for num_layers in [3, 4, 5, 6, 7]:
            for lr in [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]:
                for opt in ['adam']:
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
                                b_x = batch_x if is_emb else Variable(batch_x)
                                b_y = batch_y if is_emb else Variable(batch_y)

                                prediction = net(b_x, fbgemm=is_emb)
                                loss = loss_func(prediction, b_y)
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()

                            if epoch % 50 == 0:
                                print("******* Epoch {} *******".format(epoch))
                                prediction = net(
                                    [x.cuda() for x in train_x] if isinstance(train_x, list) else train_x.cuda(),
                                    fbgemm=is_emb,
                                ).cpu().detach().view(-1)
                                loss = loss_func(prediction, train_y.view(-1))
                                print("Training loss: {}".format(loss))

                        estimated_time = torch.exp(net(test_x.cuda(), fbgemm=is_emb)).cpu().detach().view(-1)
                        error = abs_err(estimated_time, torch.exp(test_y.cuda()).cpu().detach().view(-1))
                        histogram(error, is_abs=True)
                        print("******* Testing results *******")
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
                            torch.save(net.state_dict(), "{}/analysis/ml_predictors/{}/predictor_{}.pth".format(PM_HOME, GPU_NAME, suffix))
                            torch.save(optimizer.state_dict(), "{}/analysis/ml_predictors/{}/optimizer_{}.pth".format(PM_HOME, GPU_NAME, suffix))
                            with open('{}/analysis/ml_predictors/{}/best_config_{}.json'.format(PM_HOME, GPU_NAME, suffix), 'w') as f:
                                json.dump(best_config, f)
                        with open('{}/analysis/ml_predictors/{}/errors_{}.csv'.format(PM_HOME, GPU_NAME, suffix), 'a') as f:
                            if not os.path.exists('{}/analysis/ml_predictors/{}/errors_{}.csv'.format(PM_HOME, GPU_NAME, suffix)):
                                f.write("size,num_layers,lr,optimizer,loss_fn,GMAE,mean,std\n")
                            f.write("{},{},{},{},{},{:.4f},{:.4f},{:.4f}\n".format(size, num_layers, lr, opt, loss_func.__class__.__name__, gmae(error), error.mean(), error.std()))

                        print("Current best config is {}, with error {:.2f}%".format(best_config, min_error * 100.0))
                        if min_error < 0.04:
                            print("Satisfied. Stop searching.")
                            exit()

    print("Min gmae loss: {}".format(min_error))
    print("Best config: {}".format(best_config))
