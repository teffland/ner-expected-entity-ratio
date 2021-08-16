""" Convert data from the Ontonotes5 corpus into our jsonl format.
"""
from argparse import ArgumentParser
import os
import shutil
import json
from glob import glob
from copy import deepcopy
import regex as re
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
    parser.add_argument("indir", type=str, help="path to the lang directory")
    parser.add_argument("outdir", type=str, help="where to put the resulting jsonl files")
    parser.add_argument("--as-sentences", action="store_true", help="Break docs down to sentence level.")
    parser.add_argument("--is-complete", action="store_true", help="Mark the resulting data as complete")
    parser.add_argument("--loglevel", type=str, default="INFO")
    return parser.parse_args()


def run(args):
    logger.setLevel(logging.getLevelName(args.loglevel))

    # Read in data
    infiles = glob(os.path.join(args.indir, "annotations/**/**/**/*.name"))
    logger.info(f"Got {len(infiles)} input docs from {args.indir}")
    data_by_doc_id = dict()
    for f in tqdm(infiles, "Reading file"):
        logger.debug(f)
        lines = open(f).readlines()
        f_data = parse_lines(lines, as_sentence=args.as_sentences)
        data_by_doc_id[f_data[0]["doc_id"]] = f_data

    # Get the splits and write them out
    logger.info("Writing splits")
    proc_id = lambda line: line.strip().split("annotations/")[1]
    train_ids = {proc_id(line) for line in open(os.path.join(args.indir, "splits/train.id"))}
    dev_ids = {proc_id(line) for line in open(os.path.join(args.indir, "splits/development.id"))}
    test_ids = {proc_id(line) for line in open(os.path.join(args.indir, "splits/test.id"))}
    trainf = open(os.path.join(args.outdir, "train.jsonl"), "w")
    devf = open(os.path.join(args.outdir, "dev.jsonl"), "w")
    testf = open(os.path.join(args.outdir, "test.jsonl"), "w")
    logger.info(f"{len(data_by_doc_id)} docs extracted.")
    for doc_id, doc_data in data_by_doc_id.items():
        if doc_id in train_ids:
            for d in doc_data:
                trainf.write(f"{json.dumps(d)}\n")
        elif doc_id in dev_ids:
            for d in doc_data:
                devf.write(f"{json.dumps(d)}\n")
        elif doc_id in test_ids:
            for d in doc_data:
                testf.write(f"{json.dumps(d)}\n")
        else:
            logger.warning(f"Doc {doc_id} not in any train/dev/test split, putting into training data...")
            for d in doc_data:
                trainf.write(f"{json.dumps(d)}\n")

    trainf.close()
    devf.close()
    testf.close()


def parse_lines(lines, as_sentence=True):
    docline, sentences = lines[0], lines[1:-1]
    doc_id = re.search('DOCNO="([^@"]*)@', docline).group(1)
    data = []
    datum = dict(uid=doc_id, doc_id=doc_id, tokens=[], gold_annotations=[])
    for i, sent in enumerate(sentences):
        sent_datum = parse_sentence(sent)

        if as_sentence:
            sent_datum["uid"] = f"{doc_id}-S{i}"
            sent_datum["doc_id"] = doc_id
            logger.debug(f"Produced sentence: {sent_datum}")
            data.append(sent_datum)
        else:
            offset = len(datum["tokens"])
            datum["tokens"].extend(sent_datum["tokens"])
            for a in sent_datum["gold_annotations"]:
                a["start"] += offset
                a["end"] += offset
                assert " ".join(datum["tokens"][a["start"] : a["end"]]) == a["mention"]
            datum["gold_annotations"].extend(sent_datum["gold_annotations"])
            if offset:
                if "sentence_ends" in datum:
                    datum["sentence_ends"].append(offset)
                else:
                    datum["sentence_ends"] = [offset]
    if not as_sentence:
        offset = len(datum["tokens"])
        if "sentence_ends" in datum:
            datum["sentence_ends"].append(offset)
        else:
            datum["sentence_ends"] = [offset]
        data.append(datum)

    return data


def parse_sentence(sent):
    # Sentences look like
    # 'And <ENAMEX TYPE="PERSON" S_OFF="3">Martine</ENAMEX> knew how to handle children !\n',
    # We hack around the metadata spaces with _
    sent = sent.replace(" TYPE=", "_TYPE=").replace(" S_OFF=", "_S_OFF=").replace(" E_OFF=", "_E_OFF=")
    raw_tokens = sent.strip().split()
    tokens = []
    annotations = []
    for j, t in enumerate(raw_tokens):
        # Signal when we start an entity
        if t.startswith("<ENAMEX"):
            meta, t = t.split(">")[:2]  # TYPE="PERSON">Martine</ENAMEX> -> TYPE="PERSON", Martine</ENAMEX, ...
            ent_type = re.search('TYPE="([^"]*)"', meta).group(1)  # TYPE="PERSON" -> PERSON
            start = len(tokens)
            annotation = dict(type=ent_type, start=start, kind="entity")
            if t.endswith("</ENAMEX"):
                t += ">"

        # Finish an entity, can happen in same iteration as when we started it, but doesnt have to
        if t.endswith("</ENAMEX>"):
            t = t[: -len("</ENAMEX>")]
            if t:
                tokens.append(t)
            annotation["end"] = len(tokens)
            annotation["mention"] = " ".join(tokens[annotation["start"] : annotation["end"]])
            annotations.append(annotation)
            annotation = None
        else:
            tokens.append(t)
    return dict(tokens=tokens, gold_annotations=annotations)


if __name__ == "__main__":
    run(parse_args())
