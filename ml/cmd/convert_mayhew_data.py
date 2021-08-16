""" Convert the conll data provided by Stephen Mayhew from the CogComp format into our jsonl format.
"""
from argparse import ArgumentParser
import os
import shutil
import json
from glob import glob
from copy import deepcopy
import logging
import colorlog
from tqdm import tqdm


class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


logger = logging.getLogger(__name__)
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


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("indir", type=str)
    parser.add_argument("outpath", type=str)
    parser.add_argument("--as-sentences", action="store_true", help="Break docs down to sentence level.")
    parser.add_argument("--is-complete", action="store_true", help="Mark the resulting data as complete")
    parser.add_argument("--loglevel", type=str, default="INFO")
    return parser.parse_args()


def run(args):
    logger.setLevel(logging.getLevelName(args.loglevel))

    # Read in their data
    their_full_data = [d for d in [json.load(open(f)) for f in sorted(glob(os.path.join(args.indir, "*txt")))]]
    logger.info(f"Got {len(their_full_data)} input docs from {args.indir}")

    # Filter out only the conll ner data we want a
    # and get put it into our format, at the document level
    their_data_doc_jsonl = []
    for i, d in enumerate(their_full_data):
        tokens = d["tokens"]
        assert len([v for v in d["views"] if v["viewName"] == "NER_CONLL"]) == 1
        labels = [v for v in d["views"] if v["viewName"] == "NER_CONLL"][0]["viewData"][0].get("constituents", [])
        if len(labels) == 0:
            logger.info(f"Datum {i} has no labels for tokens: {tokens}")
        annotations_our_way = [
            dict(
                kind="entity",
                type=u["label"],
                start=u["start"],
                end=u["end"],
                mention=" ".join(tokens[u["start"] : u["end"]]),
            )
            for u in labels
        ]
        logger.debug(tokens)
        doc_jsonl = dict(
            tokens=tokens,
            gold_annotations=annotations_our_way,
            sentence_ends=d["sentences"]["sentenceEndPositions"],
            id=d["id"],
        )
        their_data_doc_jsonl.append(doc_jsonl)
        # print()
        logger.debug(f"Doc {i}: {doc_jsonl}")

    # Now, maybe break these down into sentences
    if args.as_sentences:
        logger.info("Breaking into sentences")
        their_data_jsonl = []
        for i, d in enumerate(their_data_doc_jsonl):
            sid = 1
            logger.debug(f"{i}, got doc {d}")
            s = 0
            did = int(d["id"][: -len(".txt")]) + 1
            for e in d["sentence_ends"]:
                sent = d["tokens"][s:e]
                uid = f"D{did}-S{sid}"
                sent_anns = [deepcopy(a) for a in d["gold_annotations"] if s <= a["start"] < e and s < a["end"] <= e]
                for a in sent_anns:
                    a["start"] -= s
                    a["end"] -= s

                their_data_jsonl.append(
                    dict(uid=uid, tokens=sent, gold_annotations=sent_anns, is_complete=args.is_complete)
                )
                logger.debug(f"Made sentence: { their_data_jsonl[-1]}")
                s = e
                sid += 1
        logger.info(f"Broke into {len(their_data_jsonl)} sentences")
    else:
        their_data_jsonl = their_data_doc_jsonl

    outpath = args.outpath if args.outpath.endswith(".jsonl") else args.outpath + ".jsonl"
    with open(outpath, "w") as outf:
        for d in their_data_jsonl:
            outf.write(f"{json.dumps(d)}\n")


if __name__ == "__main__":
    run(parse_args())
