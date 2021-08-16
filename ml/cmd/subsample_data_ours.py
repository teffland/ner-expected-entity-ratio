""" Given data files in standard format, subsample them at random and output the subsampled datasets, 
along with various stats.

In our paper this is called the "Exploratory Expert" regime.

If inpath = 'data/train.jsonl' and n = 1000, min_per_class = 5
then the outputs will be:
  'data/train_P-1000.jsonl'
  
P-1000 will have 1000 annotations, dropped randomly, with min_per_class of each.
"""

import json
import h5py
import torch
from time import time
from tqdm import tqdm
from argparse import ArgumentParser
from glob import glob
from collections import Counter, defaultdict

from copy import deepcopy
import numpy as np
import random

import logging
import colorlog

logger = logging.getLogger(__name__)


class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


handler = TqdmHandler()
handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(name)s | %(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%d-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "white",
            "SUCCESS:": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
)
logger.addHandler(handler)


def parse_args(args=None):
    parser = ArgumentParser()

    parser.add_argument("inpath", type=str)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--n-per-doc", type=int, default=10)
    parser.add_argument("--drop-prob", type=float, default=0.2)
    parser.add_argument("--kind", type=str, default="entity")
    parser.add_argument("--suffix", type=str, default="")

    parser.add_argument("--limit", type=int, default=None, help="Limit to first N sentences")
    parser.add_argument("--loglevel", type=str, default="INFO")

    parser.add_argument("--random-seed", type=int, default=42)

    if args:
        return parser.parse_args(args)
    else:
        return parser.parse_args()


def run(args):
    logger.setLevel(logging.getLevelName(args.loglevel))
    logger.info(f"Args: {args}")

    random.seed(args.random_seed)
    outprefix = args.inpath[: -len(".jsonl")]

    # Read in the data, limiting to the given kind
    logger.info("Loading data")
    data = [json.loads(line) for line in open(args.inpath)]
    for d in data:
        d["gold_annotations"] = [a for a in d["gold_annotations"] if a["kind"] == args.kind]

    # Randomize docs and then do subsampling until we're done
    random.shuffle(data)
    n_total = 0
    psl_data = []
    for i in range(len(data)):
        datum, n_total = sample_datum(data[i], n_total, args)
        logger.info(f"i: {i}, N total: {n_total}")
        psl_data.append(datum)
        if n_total >= args.n:
            assert n_total == args.n

            # Clear out any remaining data and add
            if (i + 1) < len(data):
                for datum in data[i + 1 :]:
                    datum["gold_annotations"] = []
                    datum["is_complete"] = False
                    psl_data.append(datum)
            break

    # Sanity checks
    assert len(data) == len(psl_data), f"{len(data)} != {len(psl_data)}"
    assert (
        sum(len(d["gold_annotations"]) for d in psl_data) == args.n
    ), f"Got sample n: {sum(len(d['gold_annotations']) for d in psl_data)}"
    if args.n_per_doc:
        assert all(len(d["gold_annotations"]) <= args.n_per_doc for d in psl_data)

    # Write out the data
    rs = f"_rs{args.random_seed}" if args.random_seed != 42 else ""
    with open(f"{outprefix}_P-{args.n}{args.suffix}{rs}.jsonl", "w") as outf:
        for d in psl_data:
            outf.write(f"{json.dumps(d)}\n")

    logger.info("All done")


def sample_datum(datum, n_total, args):
    """Sample entities from a completed datum, moving left to right, dropping some with given prob and stopping after
    we get enough.
    """
    datum["is_complete"] = False
    kept_annotations = []
    for ann in sorted(datum["gold_annotations"], key=lambda a: a["start"]):
        if (args.n_per_doc and len(kept_annotations) >= args.n_per_doc) or n_total >= args.n:
            break
        keep = random.random() > args.drop_prob
        if keep:
            kept_annotations.append(ann)
            n_total += 1
    datum["gold_annotations"] = kept_annotations
    return datum, n_total


if __name__ == "__main__":
    run(parse_args())
