import argparse
from analysis.inference import infer

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Get performance model error for ops.")
    parser.add_argument("--op-type", type=str, default="all")
    parser.add_argument("--backward", action="store_true", default=False)
    args = parser.parse_args()

    if args.op_type == "all":
        op_list = ['embedding_lookup', "fully_connected", "conv", "concat", "memcpy", "transpose", "tril"]
        pass_list = ["forward", "backward"]
    else:
        op_list = [args.op_type]
        pass_list = [args.backward]

    for op_type in op_list:
        for p in pass_list:
            if (op_type == "fully_connected" or op_type == "conv" or op_type == "transpose" or op_type == "concat" or op_type == "memcpy") and p == "backward": # No backward for these ops
                continue
            if op_type == "embedding_lookup":
                for big in [False, True]:
                    for hit_rate_estimation in [False, True]:
                        infer(op_type, p=="backward", big=big, hit_rate_estimation=hit_rate_estimation)
            else:
                infer(op_type, p=="backward")
