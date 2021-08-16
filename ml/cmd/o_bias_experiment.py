""" Run Raw+CD (Cost-aware Decoding) experiments by taking a pretrained model and performing bayesian hp search
for the best O bias.
"""

import json
import subprocess
import os
import sys
from time import time
import skopt
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-folder", type=str)
    parser.add_argument("--dev-file", type=str)
    parser.add_argument("--test-file", type=str)
    parser.add_argument("--n-calls", type=int, default=20)
    return parser.parse_args()


def run(opts):
    print("opts", opts)

    def objective(*args):
        point = args[0]
        O_cost = point[0]
        model_folder = opts.model_folder
        dev_file = opts.dev_file

        overrides = {
            "validation_dataloader": {"batch_sampler": {"batch_size": 32}},
            "model": {
                "O_cost": O_cost,
            },
        }

        metrics_file = os.path.join(model_folder, f"Oc={O_cost}_dev-metrics.json")
        overrides = json.dumps(overrides, indent=None)
        cmd = f"""allennlp evaluate\
        {model_folder}\
        {dev_file}\
        -o '{overrides}'\
        --output-file {metrics_file}\
        --cuda-device 0\
        --include ml
        """
        print("Running", O_cost, cmd, flush=True)

        t0 = time()
        proc = subprocess.run(cmd, shell=True, check=True)
        print(f"Took {time()-t0} sec")

        metrics = json.load(open(metrics_file))

        print(json.dumps(metrics, indent=2))
        return -metrics["f1-measure-overall"]

    space = [skopt.space.Real(1e-4, 1e2, "log-uniform", name="O_cost")]

    res_gp = skopt.gp_minimize(objective, space, n_calls=opts.n_calls, random_state=0)

    print(f"Best score={res_gp.fun:.4f}")
    best_O_cost = res_gp.x[0]
    print(f"Best_O_cost={best_O_cost}")
    print(f"All evals: {list(zip(res_gp.x_iters, res_gp.func_vals))}")

    overrides = {
        "validation_dataloader": {"batch_sampler": {"batch_size": 32}},
        "model": {
            "O_cost": best_O_cost,
        },
    }
    test_file = opts.test_file
    model_folder = opts.model_folder

    metrics_file = os.path.join(model_folder, f"best-dev_Oc={best_O_cost}_test-metrics.json")
    overrides = json.dumps(overrides, indent=None)
    cmd = f"""allennlp evaluate\
    {model_folder}\
    {test_file}\
    -o '{overrides}'\
    --output-file {metrics_file}\
    --cuda-device 0\
    --include ml
    """
    print("Running", best_O_cost, cmd)

    t0 = time()
    proc = subprocess.run(cmd, shell=True, check=True)
    print(f"Took {time()-t0} sec")

    metrics = json.load(open(metrics_file))

    print("Test metrics using best O_cost={best_O_cost}")
    print(json.dumps(metrics, indent=2))

    print("ALL DONE")


if __name__ == "__main__":
    run(parse_args())
