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

import argparse, glob, json
from analysis.trace_utils import create_shared_overhead
from analysis.utils import PM_HOME, GPU_NAME

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Create shared overheads for E2E prediction of all workloads.")
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()
    print("======= Creating shared overheads... =======")

    root_dir = '{}/data/{}/e2e'.format(PM_HOME, GPU_NAME)
    # Collect all overhead files except for per-process ones
    overhead_stats_files = \
        set(glob.glob("{}/*/*overhead_stats_{}.json".format(root_dir, args.iters))) - \
        set(glob.glob("{}/*/*_[0-9]_overhead_stats_{}.json".format(root_dir, args.iters)))
    overhead_raw_files = \
        set(glob.glob("{}/*/*overhead_raw_{}.csv".format(root_dir, args.iters))) - \
        set(glob.glob("{}/*/*_[0-9]_overhead_raw_{}.csv".format(root_dir, args.iters)))

    shared_overhead = create_shared_overhead(overhead_raw_files, overhead_stats_files)

    # Save to json file
    with open("{}/shared_overheads.json".format(root_dir), 'w') as f:
        json.dump(shared_overhead, f)