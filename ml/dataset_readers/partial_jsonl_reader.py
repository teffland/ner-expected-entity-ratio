from typing import Iterator, List, Dict, Any, Union, Tuple
import torch
import torch.optim as optim
import numpy as np
from allennlp.data import Instance
from allennlp.data.fields import (
    TextField,
    SequenceLabelField,
    ListField,
    LabelField,
    MetadataField,
    ArrayField,
)
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import (
    TokenIndexer,
    SingleIdTokenIndexer,
    PretrainedTransformerIndexer,
)
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary

import logging

logger = logging.getLogger(__name__)

import h5py
import numpy as np
import json
from collections import defaultdict

from ml.dataset_readers.transformers_converter import TransformersConverter


@DatasetReader.register("partial-jsonl", exist_ok=True)
class PartialJsonlReader(DatasetReader):
    """
    DatasetReader for jsonl data in standardized format:

    datum = {
      'uid': str,           # unique id for datum in the corpus (must be unique across train/dev/test)
      'tokens': List[str],  # lexical tokenization (not the same as BPE used by pretrained LMs)
      'is_complete': bool,  # whether to assume unannotated tokens are "O" or "-"
      'gold_annotations': List[Dict[str,Any]] with format:
          {
            'kind': str,    # in ('pos', 'chunk', 'entity')
            'type': str,    # the actual annotation tag/class e.g., PER
            'start': int,   # start token index
            'end': int,     # end token index
            'mention': str, # mention == ' '.join(tokens[start:end])
          }
    }

    This dataset reader will further tokenize the data into BPEs given by the provided language model
    and will extend the annotation types to a BILOU encoding over the BPEs.

    e.g.,
       given an input data like:
         tokens = ["Barack", "Obama", "was", "president"]
         gold_annotations = [ { ... 'start':0, 'end':2', 'type':'PER' ...} ]
         is_complete = False

       might be tokenized as:
         "Bar",   "ack",  "_Obama", "_was", "_president"

       and the resulting tags will be:
         "B-PER", "I-PER", "L-PER",  "-", "-"

       with token mapping:
         t2b = {
           0: [0,1],  # "Barack" -> "Bar", "ack"
           1: [2],
           2: [3],
           3: [4]
         }


    **NOTE**: Currently this reader only handles Bert and Roberta language models from transformers

    """

    def __init__(
        self,
        token_namespace: str = "tokens",
        token_indexers: Dict[str, TokenIndexer] = None,
        model_name: str = None,
        assume_complete: Union[str, bool] = False,
        latent_tag: str = "_",
        O_tag: str = "O",
        kind: str = "entity",
        label_encoding: str = "BIOUL",
        drop_unannotated_sentences: bool = False,
        drop_unannotated_docs: bool = False,
        debug: bool = False,
        lazy: bool = False,
        limit: int = None,
    ) -> None:
        """
        Args:
          token_namespace: key for tokens field
          token_indexers: set of indexers for tokens
          model_name: pretrained lm to use for subword tokenization
          assume_complete: whether to assume data are complete if they are missing `is_complete` field.
          latent_tag: special tag to use as "latent variable" signal to model.
          O_tag: special tag to use as "filler" in between annotations.
          kind: subtype of annotations to extract (for our experiments it's always "entity")
          label_encoding: Should use BIOUL
          drop_unannotated_sentences: drop all unannotated sentences after the last annotated sentence in a doc. This
            is the "shortest" preprocessing setting in the paper.
          drop_unannotated_doxs: drop docs with no annotations. This is the "short" preprocessing setting in the paper.
          debug: whether to show debug outputs
          lazy: whether to load in all data at once.
          limit: limit number of instances for debugging
        """
        super().__init__(lazy=lazy)
        self.limit = limit
        self.token_namespace = token_namespace
        self.subword_converter = TransformersConverter(model_name) if model_name else None
        self.maxlen = self.subword_converter.tokenizer.model_max_length if model_name else None

        if not token_indexers:
            if model_name:
                token_indexers = {
                    "tokens": PretrainedTransformerIndexer(
                        model_name,
                        namespace=token_namespace,
                        tokenizer_kwargs=dict(use_fast=True),
                    )
                }
            else:
                token_indexers = {"tokens": SingleIdTokenIndexer(token_namespace)}
        self.token_indexers = token_indexers

        if isinstance(assume_complete, str):
            assume_complete = assume_complete.lower() == "true"
        self.assume_complete = assume_complete
        self.latent_tag = "_"
        self.O = O_tag
        self.label_encoding = label_encoding
        self.kind = kind

        assert not (drop_unannotated_docs and drop_unannotated_sentences)
        self.drop_unannotated_sentences = drop_unannotated_sentences
        self.drop_unannotated_docs = drop_unannotated_docs

        if debug:
            logger.setLevel(logging.DEBUG)

    def text_to_instances(self, tokens: List[str], annotations: List[Dict[str, Any]] = [], **metadata) -> Instance:
        metadata["og_tokens"] = tokens
        if self.subword_converter is not None:
            tokens, tokidx2bpeidxs = self.subword_converter(tokens)
        else:
            tokidx2bpeidxs = {i: [i] for i in range(len(tokens))}
        metadata["tokidx2bpeidxs"] = tokidx2bpeidxs
        tags = self.get_tags(tokens, annotations, metadata)
        # print("go;d tags", tags)

        for tokens, tags, metadata in self.as_maximal_subdocs(tokens, tags, metadata):
            metadata["bpe_tokens"] = tokens
            tokens = [Token(t) for t in tokens]
            tokens_field = TextField(tokens, self.token_indexers)
            tag_namespace = "labels"
            fields = dict(
                tokens=tokens_field,
                tags=SequenceLabelField(tags, tokens_field, label_namespace=tag_namespace),
                metadata=MetadataField(metadata),
            )

            yield Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        og_n, actual_n = 0, 0
        with open(file_path) as f:
            for i, line in enumerate(f):
                if self.limit and i >= self.limit:
                    return
                datum = json.loads(line)
                tokens = datum.pop("tokens")
                annotations = datum.pop("gold_annotations")

                if self.drop_unannotated_sentences:
                    n = len(tokens)
                    og_n += n
                    tokens = self._get_annotated_subdoc(tokens, annotations, datum)
                    actual_n += len(tokens)
                    print(f"{i} Dropped from {n} tokens to {len(tokens)} annotated ones", flush=True)
                elif self.drop_unannotated_docs:
                    og_n += 1
                    if not annotations:
                        tokens = []
                        print(f"{i} Dropped unannotated doc", flush=True)
                    else:
                        actual_n += 1

                if tokens:
                    for instance in self.text_to_instances(tokens=tokens, annotations=annotations, **datum):
                        yield instance

        if self.drop_unannotated_sentences:
            print(f"Cut down total tokens from {og_n} to {actual_n} = {100.*actual_n/og_n} %", flush=True)
        elif self.drop_unannotated_docs:
            print(f"Cut down docs from {og_n} to {actual_n} = {100.*actual_n/og_n} %", flush=True)

    def get_tags(
        self,
        tokens: List[str],
        annotations: Dict[str, Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> List[str]:
        """Create tag sequence from annotations and possible subword mapping."""
        # Filter down annotations only to the specified kind
        annotations = [a for a in annotations if a["kind"] == self.kind]

        tokidx2bpeidxs = metadata["tokidx2bpeidxs"]
        n = len(tokens)
        # print("tokens", tokens)
        # Default tags are either O or latent
        if self.assume_complete:
            tags = [self.O] * n
        else:
            tags = [self.latent_tag] * n
        # Fill in any partial annotations
        # Map start,ends for the observed annotations onto tokens
        # print("---")
        # print(metadata)
        # print(n, tokens)
        # print(annotations)
        # print(tokidx2bpeidxs)
        for ann in annotations:
            s, e, t = (
                tokidx2bpeidxs[ann["start"]][0],
                tokidx2bpeidxs[ann["end"] - 1][-1],
                ann["type"],
            )
            # print(s, e, t, ann)
            if t == self.O:
                for k in range(s, e + 1):
                    tags[k] = self.O
            else:
                if self.label_encoding == "BIOUL":
                    if e - s > 0:
                        tags[s] = f"B-{t}"
                        for k in range(s + 1, e):
                            tags[k] = f"I-{t}"
                        tags[e] = f"L-{t}"
                    else:
                        tags[s] = f"U-{t}"
                elif self.label_encoding == "BIO":
                    tags[s] = f"B-{t}"
                    if e - s > 0:
                        for k in range(s + 1, e + 1):
                            tags[k] = f"I-{t}"
                else:
                    raise ValueError(self.label_encoding)

        # If we are using transfomers, force first and last tokens (specials tokens) to be O tag
        if self.subword_converter is not None:
            tags[0] = self.O
            tags[-1] = self.O

        # print(tags, flush=True)

        return tags

    def as_maximal_subdocs(
        self,
        tokens: List[str],
        tags: List[str],
        metadata: Dict[str, Any],
    ) -> List[Tuple[List[str], List[str], Dict[str, Any]]]:
        """Break up docuemnts that are too large along sentence boundaries into ones that fit within the length limit."""
        if self.maxlen is None or len(tokens) <= self.maxlen:
            subdocs = [(tokens, tags, metadata)]
        else:
            subdocs = []
            tok2bpes = metadata.pop("tokidx2bpeidxs")
            bpe2tok = {v: k for k, vs in tok2bpes.items() for v in vs}
            uid = metadata.pop("uid", metadata.get("id", "NO ID"))
            # print(f"Breaking up sentences for {uid} with len: {len(tokens)}")
            s_tok, s_bpe, s_L = 0, 0, 0
            if "sentence_ends" in metadata:
                ends = metadata.pop("sentence_ends")
            else:
                print(f"No sentence ends found for {uid}, using crude length-based boundaries")
                ends = list(range(1, len(tok2bpes) + 1))
            subdoc_tokens, subdoc_tags, subdoc_metadata = None, None, None

            # We use a slightly smaller than actually ok subgrouping length because sometimes breaking up the sentences
            # that are too long starts or ends in the middle of a word and fixing to the full word goes over the maxlen
            maxlen = self.maxlen - 10
            for i, e_tok in enumerate(ends):
                # print(f"i:{i}, e:{e_tok}, bpes:{tok2bpes.keys()}")
                e_bpe = tok2bpes[e_tok][0] if e_tok < len(tok2bpes) else len(tokens)

                # Check to see if this sentence would put the current subdoc over the edge.
                if i and (e_bpe - s_bpe) > maxlen:
                    # If so, finish off the subdoc and advance the start cursors
                    subdoc_tok2bpe = {(k - s_tok): [v - s_bpe for v in tok2bpes[k]] for k in range(s_tok, e_tok)}
                    subdoc_metadata = dict(
                        uid=f"{uid}-S{s_L}:{i-1}",
                        tokidx2bpeidxs=subdoc_tok2bpe,
                        **metadata,
                    )
                    logger.debug(f"\nAdding subdoc {subdoc_metadata['uid']} with len {len(subdoc_tokens)}")
                    logger.debug(
                        f'{" ".join([f"{t}/{l}" if l != self.latent_tag else t for t, l in zip(subdoc_tokens, subdoc_tags)])}'
                    )
                    assert len(subdoc_tokens) == len(subdoc_tags)
                    subdocs.append((subdoc_tokens, subdoc_tags, subdoc_metadata))
                    s_tok = ends[i - 1]
                    s_bpe = tok2bpes[s_tok][0]
                    s_L = i

                # Compute the next candidate subdoc
                # If the the next candidate subdoc will be too long on its own, break it up into smaller pieces that fit.
                # (ie, when there is sentence that is too long)
                if (e_bpe - s_bpe) > maxlen:
                    # Make sure the subgroups start/end on word boundaries
                    def to_word_boundary(bpe):
                        if bpe in (0, len(tokens)):
                            return bpe
                        else:
                            return tok2bpes[bpe2tok[bpe]][0]

                    n_groups = int(np.ceil((e_bpe - s_bpe) / maxlen))
                    s_bpes = [to_word_boundary(s_bpe + g * maxlen) for g in range(n_groups)]
                    e_bpes = [to_word_boundary(min(e_bpe, s_bpe + (g + 1) * maxlen)) for g in range(n_groups)]
                else:
                    s_bpes = [s_bpe]
                    e_bpes = [e_bpe]
                for g, (s_bpe, e_bpe) in enumerate(zip(s_bpes, e_bpes)):
                    # Collect the sentence tokens, tags, and add on start/end tokens&tags where needed
                    subdoc_tokens = tokens[s_bpe:e_bpe]
                    subdoc_tags = tags[s_bpe:e_bpe]
                    if not subdoc_tokens[0] == tokens[0]:
                        subdoc_tokens = [tokens[0]] + subdoc_tokens
                        subdoc_tags = [tags[0]] + subdoc_tags
                    if not subdoc_tokens[-1] == tokens[-1]:
                        subdoc_tokens = subdoc_tokens + [tokens[-1]]
                        subdoc_tags = subdoc_tags + [tags[-1]]

                    if g < len(s_bpes) - 1:
                        # All but the last group get turned into subdocs here
                        e_tok = bpe2tok[e_bpe]
                        subdoc_tok2bpe = {(k - s_tok): [v - s_bpe for v in tok2bpes[k]] for k in range(s_tok, e_tok)}
                        subdoc_metadata = dict(
                            uid=f"{uid}-S{s_L}:{i-1}.G{g}",
                            tokidx2bpeidxs=subdoc_tok2bpe,
                            **metadata,
                        )
                        logger.debug(f"\nAdding subdoc {subdoc_metadata['uid']} with len {len(subdoc_tokens)}")
                        logger.debug(
                            f'{" ".join([f"{t}/{l}" if l != self.latent_tag else t for t, l in zip(subdoc_tokens, subdoc_tags)])}'
                        )
                        assert len(subdoc_tokens) == len(subdoc_tags)
                        subdocs.append((subdoc_tokens, subdoc_tags, subdoc_metadata))
                    s_tok = e_tok

            # Add the last one
            subdoc_tok2bpe = {(k - s_tok): [v - s_bpe for v in tok2bpes[k]] for k in range(s_tok, e_tok)}
            subdoc_metadata = dict(uid=f"{uid}-S{s_L}:{i}", tokidx2bpeidxs=subdoc_tok2bpe, **metadata)
            # print(f"\nAdding subdoc {subdoc_metadata['uid']} with len {len(subdoc_tokens)}")
            # print(" ".join([f"{t}/{l}" if l != self.latent_tag else t for t, l in zip(subdoc_tokens, subdoc_tags)]))
            subdocs.append((subdoc_tokens, subdoc_tags, subdoc_metadata))

            cat_tokens = [t for (subdoctoks, _, _) in subdocs for t in subdoctoks[1:-1]]
            assert len(cat_tokens) == len(tokens) - 2, f"{len(cat_tokens)} != {len(tokens)-2}"
            assert cat_tokens == tokens[1:-1], f"{list(zip(cat_tokens, tokens[1:-1]))}"
        return subdocs

    def _get_annotated_subdoc(self, tokens, annotations, metadata):
        """ Chop off trailing sentences where there are no annotations. """
        annotations = [a for a in annotations if a["kind"] == self.kind]
        if annotations:
            last_ann = sorted(annotations, key=lambda a: -a["end"])[0]
            ends = sorted([e for e in metadata["sentence_ends"] if e >= last_ann["end"]])
            if ends:
                return tokens[: ends[0]]
            else:
                return tokens
        else:
            return []


def test():
    data = [
        {
            "uid": "eng.testa-D1-S1",
            "tokens": [
                "CRICKET",
                "-",
                "LEICESTERSHIRE",
                "TAKE",
                "OVER",
                "AT",
                "TOP",
                "AFTER",
                "INNINGS",
                "VICTORY",
                ".",
            ],
            "gold_annotations": [
                {
                    "kind": "pos",
                    "type": "NNP",
                    "start": 0,
                    "end": 1,
                    "mention": "CRICKET",
                },
                {"kind": "pos", "type": ":", "start": 1, "end": 2, "mention": "-"},
                {
                    "kind": "chunk",
                    "type": "NP",
                    "start": 0,
                    "end": 1,
                    "mention": "CRICKET",
                },
                {
                    "kind": "pos",
                    "type": "NNP",
                    "start": 2,
                    "end": 3,
                    "mention": "LEICESTERSHIRE",
                },
                {"kind": "pos", "type": "NNP", "start": 3, "end": 4, "mention": "TAKE"},
                {
                    "kind": "entity",
                    "type": "ORG",
                    "start": 2,
                    "end": 3,
                    "mention": "LEICESTERSHIRE",
                },
                {"kind": "pos", "type": "IN", "start": 4, "end": 5, "mention": "OVER"},
                {
                    "kind": "chunk",
                    "type": "NP",
                    "start": 2,
                    "end": 4,
                    "mention": "LEICESTERSHIRE TAKE",
                },
                {"kind": "pos", "type": "NNP", "start": 5, "end": 6, "mention": "AT"},
                {
                    "kind": "chunk",
                    "type": "PP",
                    "start": 4,
                    "end": 5,
                    "mention": "OVER",
                },
                {"kind": "pos", "type": "NNP", "start": 6, "end": 7, "mention": "TOP"},
                {
                    "kind": "pos",
                    "type": "NNP",
                    "start": 7,
                    "end": 8,
                    "mention": "AFTER",
                },
                {
                    "kind": "pos",
                    "type": "NNP",
                    "start": 8,
                    "end": 9,
                    "mention": "INNINGS",
                },
                {
                    "kind": "pos",
                    "type": "NN",
                    "start": 9,
                    "end": 10,
                    "mention": "VICTORY",
                },
                {"kind": "pos", "type": ".", "start": 10, "end": 11, "mention": "."},
                {
                    "kind": "chunk",
                    "type": "NP",
                    "start": 5,
                    "end": 10,
                    "mention": "AT TOP AFTER INNINGS VICTORY",
                },
            ],
            "is_complete": True,
        },
        {
            "uid": "eng.testa-D1-S2",
            "tokens": ["LONDON", "1996-08-30"],
            "gold_annotations": [
                {
                    "kind": "pos",
                    "type": "NNP",
                    "start": 0,
                    "end": 1,
                    "mention": "LONDON",
                },
                {
                    "kind": "pos",
                    "type": "CD",
                    "start": 1,
                    "end": 2,
                    "mention": "1996-08-30",
                },
                {
                    "kind": "entity",
                    "type": "LOC",
                    "start": 0,
                    "end": 1,
                    "mention": "LONDON",
                },
                {
                    "kind": "chunk",
                    "type": "NP",
                    "start": 0,
                    "end": 2,
                    "mention": "LONDON 1996-08-30",
                },
            ],
            "is_complete": False,
        },
        {
            "uid": "eng.testa-D1-S3",
            "tokens": [
                "West",
                "Indian",
                "all-rounder",
                "Phil",
                "Simmons",
                "took",
                "four",
                "for",
                "38",
                "on",
                "Friday",
                "as",
                "Leicestershire",
                "beat",
                "Somerset",
                "by",
                "an",
                "innings",
                "and",
                "39",
                "runs",
                "in",
                "two",
                "days",
                "to",
                "take",
                "over",
                "at",
                "the",
                "head",
                "of",
                "the",
                "county",
                "championship",
                ".",
            ],
            "gold_annotations": [
                {"kind": "pos", "type": "NNP", "start": 0, "end": 1, "mention": "West"},
                {
                    "kind": "pos",
                    "type": "NNP",
                    "start": 1,
                    "end": 2,
                    "mention": "Indian",
                },
                {
                    "kind": "pos",
                    "type": "NN",
                    "start": 2,
                    "end": 3,
                    "mention": "all-rounder",
                },
                {
                    "kind": "entity",
                    "type": "MISC",
                    "start": 0,
                    "end": 2,
                    "mention": "West Indian",
                },
                {"kind": "pos", "type": "NNP", "start": 3, "end": 4, "mention": "Phil"},
                {
                    "kind": "pos",
                    "type": "NNP",
                    "start": 4,
                    "end": 5,
                    "mention": "Simmons",
                },
                {"kind": "pos", "type": "VBD", "start": 5, "end": 6, "mention": "took"},
                {
                    "kind": "chunk",
                    "type": "NP",
                    "start": 0,
                    "end": 5,
                    "mention": "West Indian all-rounder Phil Simmons",
                },
                {
                    "kind": "entity",
                    "type": "PER",
                    "start": 3,
                    "end": 5,
                    "mention": "Phil Simmons",
                },
                {"kind": "pos", "type": "CD", "start": 6, "end": 7, "mention": "four"},
                {
                    "kind": "chunk",
                    "type": "VP",
                    "start": 5,
                    "end": 6,
                    "mention": "took",
                },
                {"kind": "pos", "type": "IN", "start": 7, "end": 8, "mention": "for"},
                {
                    "kind": "chunk",
                    "type": "NP",
                    "start": 6,
                    "end": 7,
                    "mention": "four",
                },
                {"kind": "pos", "type": "CD", "start": 8, "end": 9, "mention": "38"},
                {"kind": "chunk", "type": "PP", "start": 7, "end": 8, "mention": "for"},
                {"kind": "pos", "type": "IN", "start": 9, "end": 10, "mention": "on"},
                {"kind": "chunk", "type": "NP", "start": 8, "end": 9, "mention": "38"},
                {
                    "kind": "pos",
                    "type": "NNP",
                    "start": 10,
                    "end": 11,
                    "mention": "Friday",
                },
                {"kind": "chunk", "type": "PP", "start": 9, "end": 10, "mention": "on"},
                {"kind": "pos", "type": "IN", "start": 11, "end": 12, "mention": "as"},
                {
                    "kind": "chunk",
                    "type": "NP",
                    "start": 10,
                    "end": 11,
                    "mention": "Friday",
                },
                {
                    "kind": "pos",
                    "type": "NNP",
                    "start": 12,
                    "end": 13,
                    "mention": "Leicestershire",
                },
                {
                    "kind": "chunk",
                    "type": "PP",
                    "start": 11,
                    "end": 12,
                    "mention": "as",
                },
                {
                    "kind": "pos",
                    "type": "VBD",
                    "start": 13,
                    "end": 14,
                    "mention": "beat",
                },
                {
                    "kind": "chunk",
                    "type": "NP",
                    "start": 12,
                    "end": 13,
                    "mention": "Leicestershire",
                },
                {
                    "kind": "entity",
                    "type": "ORG",
                    "start": 12,
                    "end": 13,
                    "mention": "Leicestershire",
                },
                {
                    "kind": "pos",
                    "type": "NNP",
                    "start": 14,
                    "end": 15,
                    "mention": "Somerset",
                },
                {
                    "kind": "chunk",
                    "type": "VP",
                    "start": 13,
                    "end": 14,
                    "mention": "beat",
                },
                {"kind": "pos", "type": "IN", "start": 15, "end": 16, "mention": "by"},
                {
                    "kind": "chunk",
                    "type": "NP",
                    "start": 14,
                    "end": 15,
                    "mention": "Somerset",
                },
                {
                    "kind": "entity",
                    "type": "ORG",
                    "start": 14,
                    "end": 15,
                    "mention": "Somerset",
                },
                {"kind": "pos", "type": "DT", "start": 16, "end": 17, "mention": "an"},
                {
                    "kind": "chunk",
                    "type": "PP",
                    "start": 15,
                    "end": 16,
                    "mention": "by",
                },
                {
                    "kind": "pos",
                    "type": "NN",
                    "start": 17,
                    "end": 18,
                    "mention": "innings",
                },
                {"kind": "pos", "type": "CC", "start": 18, "end": 19, "mention": "and"},
                {
                    "kind": "chunk",
                    "type": "NP",
                    "start": 16,
                    "end": 18,
                    "mention": "an innings",
                },
                {"kind": "pos", "type": "CD", "start": 19, "end": 20, "mention": "39"},
                {
                    "kind": "pos",
                    "type": "NNS",
                    "start": 20,
                    "end": 21,
                    "mention": "runs",
                },
                {"kind": "pos", "type": "IN", "start": 21, "end": 22, "mention": "in"},
                {
                    "kind": "chunk",
                    "type": "NP",
                    "start": 19,
                    "end": 21,
                    "mention": "39 runs",
                },
                {"kind": "pos", "type": "CD", "start": 22, "end": 23, "mention": "two"},
                {
                    "kind": "chunk",
                    "type": "PP",
                    "start": 21,
                    "end": 22,
                    "mention": "in",
                },
                {
                    "kind": "pos",
                    "type": "NNS",
                    "start": 23,
                    "end": 24,
                    "mention": "days",
                },
                {"kind": "pos", "type": "TO", "start": 24, "end": 25, "mention": "to"},
                {
                    "kind": "chunk",
                    "type": "NP",
                    "start": 22,
                    "end": 24,
                    "mention": "two days",
                },
                {
                    "kind": "pos",
                    "type": "VB",
                    "start": 25,
                    "end": 26,
                    "mention": "take",
                },
                {
                    "kind": "pos",
                    "type": "IN",
                    "start": 26,
                    "end": 27,
                    "mention": "over",
                },
                {
                    "kind": "chunk",
                    "type": "VP",
                    "start": 24,
                    "end": 26,
                    "mention": "to take",
                },
                {"kind": "pos", "type": "IN", "start": 27, "end": 28, "mention": "at"},
                {"kind": "pos", "type": "DT", "start": 28, "end": 29, "mention": "the"},
                {
                    "kind": "chunk",
                    "type": "PP",
                    "start": 26,
                    "end": 28,
                    "mention": "over at",
                },
                {
                    "kind": "pos",
                    "type": "NN",
                    "start": 29,
                    "end": 30,
                    "mention": "head",
                },
                {"kind": "pos", "type": "IN", "start": 30, "end": 31, "mention": "of"},
                {
                    "kind": "chunk",
                    "type": "NP",
                    "start": 28,
                    "end": 30,
                    "mention": "the head",
                },
                {"kind": "pos", "type": "DT", "start": 31, "end": 32, "mention": "the"},
                {
                    "kind": "chunk",
                    "type": "PP",
                    "start": 30,
                    "end": 31,
                    "mention": "of",
                },
                {
                    "kind": "pos",
                    "type": "NN",
                    "start": 32,
                    "end": 33,
                    "mention": "county",
                },
                {
                    "kind": "pos",
                    "type": "NN",
                    "start": 33,
                    "end": 34,
                    "mention": "championship",
                },
                {"kind": "pos", "type": ".", "start": 34, "end": 35, "mention": "."},
                {
                    "kind": "chunk",
                    "type": "NP",
                    "start": 31,
                    "end": 34,
                    "mention": "the county championship",
                },
            ],
            "is_complete": False,
        },
    ]

    reader = PartialJsonlReader(model_name="roberta-base")

    for i, datum in enumerate(data):
        print(f"=== {i} ===")
        tokens = datum.pop("tokens")
        annotations = datum.pop("gold_annotations")
        instance = list(reader.text_to_instances(tokens=tokens, annotations=annotations, **datum))[0]
        tokens = [t.text for t in instance.fields["tokens"]]
        tags = [t for t in instance.fields["tags"].labels]
        print(" ".join(f"{token}/{tag}" for token, tag in zip(tokens, tags)))
        # print(instance.fields["tokens"])
        # print(instance.fields["tags"])


# test()