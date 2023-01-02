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

import argparse
from analysis.inference import infer

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Get performance model error for ops.')
    parser.add_argument('--op-type', type=str, default='all')
    parser.add_argument('--backward', action='store_true', default=False)
    parser.add_argument('--emb-heuristic', action='store_true', default=False)
    parser.add_argument("--emb-data-path-suffix", type=str, default='fbgemm_dlrm_datasets')
    parser.add_argument("--emb-table-configs-path", type=str, default='/nvme/deep-learning/dlrm_datasets/embedding_bag/2021/fbgemm_t856_bs65536_configs.json')
    args = parser.parse_args()

    if args.op_type == 'all':
        op_list = ['embedding_lookup', 'fully_connected', 'conv2d', 'conv1d', 'concat', 'memcpy', 'transpose', 'bn', 'tril']
        pass_list = ['forward', 'backward']
    else:
        op_list = [args.op_type]
        pass_list = ['backward' if args.backward else 'forward']

    for op_type in op_list:
        for p in pass_list:
            if (op_type == 'fully_connected' or \
                    op_type == 'transpose' or \
                    op_type == 'concat' or \
                    op_type == 'memcpy') and \
                        p == 'backward': # No backward for these ops
                continue
            if op_type == 'embedding_lookup' and args.emb_heuristic:
                for big in [False, True]:
                    for hit_rate_estimation in [False, True]:
                        for fbgemm in [False, True]:
                            infer(op_type, p=='backward', big=big, hit_rate_estimation=hit_rate_estimation, fbgemm=fbgemm)
            else:
                infer(
                    op_type,
                    backward=(p=='backward'),
                    emb_use_mlp=True,
                    suffix=args.emb_data_path_suffix,
                    table_configs=args.emb_table_configs_path
                )
