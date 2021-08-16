""" Used by parital_jsonl_reader to convert tokens to subword sequences and provide alignment information between the
two, so that we can also align the tags appropriately.
"""


from typing import *
from collections import defaultdict
from transformers import AutoTokenizer


class TransformersConverter(object):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def __call__(self, tokens: List[str]):
        """Tokenize and align the token indices to the subword units."""
        text = " ".join(tokens)  # we have to join them otherwise RoBERTa will break. An idiosyncrasy of 3.5.1
        tokenized = self.tokenizer(
            text, is_split_into_words=False, return_offsets_mapping=True, add_special_tokens=True
        )
        bpe_tokens = self.tokenizer.convert_ids_to_tokens(tokenized["input_ids"])
        tokidx2bpeidx = self.align_bpes(tokens, tokenized["offset_mapping"])
        return bpe_tokens, tokidx2bpeidx

    def align_bpes(self, tokens: List[str], offset_mapping: List[Tuple[int, int]]):
        """ Return a multi-mapping from token indices to subword indices"""

        # Mapping from char idxs in joined text str to original tokens
        charidx2tokidx = {}
        for i, t in enumerate(tokens):
            offset = len(charidx2tokidx) + i
            for k in range(len(t)):
                charidx2tokidx[offset + k] = i

        # Mapping derived from the offset mapping to go brom token idx to bpe idx
        tokidx2bpeidx = defaultdict(list)
        for i, (s, e) in enumerate(offset_mapping):
            if s == e:
                continue  # skip special tokens
            else:
                tokidx = charidx2tokidx[s]
                tokidx2bpeidx[tokidx].append(i)
        return tokidx2bpeidx


def test():
    sentences = ["This is a normal sentence".split(), "This is a sentence withintentionalweirdsubwords".split()]
    converter = TransformersConverter("roberta-large")
    for i, s in enumerate(sentences):
        print(f"=== {i} ===")
        print(s)
        bpes, tok2bpe = converter(s)
        print(bpes)
        print(tok2bpe)
        for tokidx, bpeidxs in tok2bpe.items():
            print(s[tokidx], bpes[bpeidxs[0] : bpeidxs[-1] + 1])


# test()