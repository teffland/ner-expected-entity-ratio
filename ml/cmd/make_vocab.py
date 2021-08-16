""" Take a dataset file and a transformer model name and pre-initialize a vocab dir for allennlp. """

from argparse import ArgumentParser
import os
import shutil

from ml.dataset_readers.partial_jsonl_reader import PartialJsonlReader


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model", type=str, help="transformer model name")
    parser.add_argument("data", type=str, help="path to test file in jsonl format")
    parser.add_argument("outpath", type=str, help="path to output location")
    parser.add_argument("--O-tag", type=str, default="O")
    parser.add_argument("--latent-tag", type=str, default="_")
    parser.add_argument("--extra-tags", type=str, nargs="+")
    return parser.parse_args()


def run(args):
    # Setup dir
    if os.path.exists(args.outpath):
        print(f"Overwriting out path: {args.outpath}")
        shutil.rmtree(args.outpath)

    os.makedirs(args.outpath)

    # Read in data
    reader = PartialJsonlReader(model_name=args.model)
    data = list(reader.read(args.data))
    print("~" * 50)

    # Output tokens
    tokenizer = reader.subword_converter.tokenizer
    vocab = sorted(list(tokenizer.get_vocab().items()), key=lambda x: x[1])
    with open(f"{args.outpath}/tokens.txt", "w") as f:
        for t, i in vocab:
            f.write(f"{t}\n")

    print(f"pad token is {tokenizer.pad_token} at {tokenizer.pad_token_id}")
    print(f"unk token is {tokenizer.unk_token} at {tokenizer.unk_token_id}")
    print(f"bos token is {tokenizer.bos_token} at {tokenizer.bos_token_id}")
    print(f"eos token is {tokenizer.eos_token} at {tokenizer.eos_token_id}")

    # Output tags
    # They are always laid out [ latent, O, ...rest] because models assume this ordering
    special_tags = (args.latent_tag, args.O_tag)
    tagset = {tag for d in data for tag in d.fields["tags"].labels if tag not in special_tags} | set(
        args.extra_tags or []
    )
    tags = list(special_tags) + sorted(list(tagset))
    print(f"Got tags {tags}")
    with open(f"{args.outpath}/labels.txt", "w") as f:
        for t in tags:
            f.write(f"{t}\n")

    with open(f"{args.outpath}/non_padded_namespaces.txt", "w") as f:
        f.write(f"*tags\n*labels\n")


if __name__ == "__main__":
    run(parse_args())
