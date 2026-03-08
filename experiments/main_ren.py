#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
The main script for running a single replication.
"""
import os
import sys
sys.path.insert(0, '/root/morbo')
from morbo.run_one_replication_ren import run_one_replication_ren
import json
import torch
import errno
from typing import Any, Dict

from chemprop.args import PredictArgs, TrainArgs, MorboArgs

def fetch_data(kwargs: Dict[str, Any]) -> None:
    # this modifies kwargs in place
    problem_kwargs = kwargs.get("problem_kwargs", {})
    key = problem_kwargs.get("datapath")

    if key is not None:
        data = torch.load(key)
        problem_kwargs["data"] = data
        kwargs["problem_kwargs"] = problem_kwargs


if __name__ == "__main__":
    kwargs = MorboArgs().parse_args()
    for seed in kwargs.seeds:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        exp_dir = os.path.join(current_dir, kwargs.exp_dir)
        # config_path = os.path.join(exp_dir, "config.json")
        # label = sys.argv[2]
        label = kwargs.label
        print(f"label:{label}")
        # seed = int(float(sys.argv[3]))
        # seed = int(float(kwargs.seed))
        print(f"seed:{seed}")
        # max_evals = int(sys.argv[4])
        max_evals = int(kwargs.max_evals)

        # last_arg = sys.argv[5] if len(sys.argv) > 5 else None
        output_path = os.path.join(exp_dir, label, f"{str(seed).zfill(4)}_{label}_{max_evals}.pt")
        print(f"output_path:{output_path}")
        if not os.path.exists(os.path.dirname(output_path)):
            try:
                os.makedirs(os.path.dirname(output_path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        # with open(config_path, "r") as f:
        #     kwargs = json.load(f)

        save_callback = lambda data: torch.save(data, output_path)
        # fetch_data(kwargs=kwargs)
        run_one_replication_ren(
            seed=seed,
            label=kwargs.label,
            save_callback=save_callback,
            max_evals=kwargs.max_evals,
            dim=kwargs.dim,
            evalfn=kwargs.evalfn,
            n_initial_points=kwargs.n_initial_points,
            batch_size=kwargs.batch_size,
            min_tr_size=kwargs.min_tr_size,
            max_reference_point=kwargs.max_reference_point,
            verbose=kwargs.verbose

        )
