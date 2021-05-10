import torch
import torch.utils.data as Data
from torch.autograd import Variable
import argparse, json, os
from analysis.utils import *
from analysis.inference import infer

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training MLP performance model for FC, transpose, and tril.")
    parser.add_argument("--op-type", type=str, required=True)
    parser.add_argument("--backward", action="store_true", default=False) # For tril
    parser.add_argument("--inference", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=800)
    args = parser.parse_args()

    assert args.op_type in ["fully_connected", "transpose", "tril"]
    suffix = "{}_{}".format(args.op_type, 1 if not args.backward else 0)
    n_feature, x, y = get_data(op_type=args.op_type, backward=args.backward)
    op_dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset=op_dataset,
        batch_size=args.batch_size,
        shuffle=True, num_workers=0)
    print("Op type: {}, dataset length: {}, batch size: {}, epoch: {}".format(args.op_type, y.shape[0], args.batch_size, args.epoch))

    suffix = "{}_{}".format(args.op_type, 1 if not args.backward else 0)
    if os.path.exists("{}/analysis/ml_predictors/{}/best_config_{}.json".format(PM_HOME, GPU_NAME, suffix)):
        best_config, min_error = infer(args.op_type, args.backward)
        if args.inference:
            exit()
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
