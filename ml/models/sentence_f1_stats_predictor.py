from typing import List, Dict

from overrides import overrides
import numpy

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import FlagField, TextField, SequenceLabelField
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

from copy import deepcopy
from allennlp.data.dataset_readers.dataset_utils.span_utils import bioul_tags_to_spans
from collections import defaultdict


@Predictor.register("sentence_f1_stats_predictor", exist_ok=True)
class SentenceF1StatsPredictor(Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a single set of tags for it.  In particular, it can be used with
    the [`CrfTagger`](https://docs.allennlp.org/models/main/models/tagging/models/crf_tagger/)
    model and also the [`SimpleTagger`](../models/simple_tagger.md) model.
    Registered as a `Predictor` with name "sentence_tagger".
    """

    def prediction_metrics(self, datum) -> JsonDict:
        gtags, ptags = [], []
        for instance in self._json_to_instances(datum):
            prediction = self.predict_instance(instance)
            gtags.extend(prediction['tags'][1:-1])
            ptags.extend(prediction['pred_tags'][1:-1])

        
        predicted_spans = bioul_tags_to_spans(ptags)
        gold_spans = bioul_tags_to_spans(gtags)
        tps, fps, fns = defaultdict(int), defaultdict(int), defaultdict(int)
        for span in predicted_spans:
            if span in gold_spans:
                tps[span[0]] += 1
                gold_spans.remove(span)
            else:
                fps[span[0]] += 1
        # These spans weren't predicted.
        for span in gold_spans:
            fns[span[0]] += 1
            
        tp, fp, fn = (sum(tps.values()), sum(fps.values()), sum(fns.values()))
                    
        return {'micro-tp': tp, 'micro-fp':fp, 'micro-fn':fn}

    def _json_to_instances(self, json_dict: JsonDict) -> Instance:
        if "gold_annotations" in json_dict:
            json_dict["annotations"] = json_dict.pop("gold_annotations", json_dict.get("annotations", []))
        return list(self._dataset_reader.text_to_instances(**json_dict))

    def get_sentence_metrics(self, prediction):
        sents_tags, sents_pred_tags = [], []
        tok2bpes = prediction['metadata']['tokidx2bpeidxs']
        sent_metric_vec = []
        s = 0
        for tok_e in prediction['metadata']['sentence_ends']:
            e = tok2bpes[tok_e-1][-1]+1
            # print(prediction['metadata']['bpe_tokens'][s:e])
            sent_tags = prediction['tags'][s:e]
            sent_pred_tags = prediction['pred_tags'][s:e]
            
            # Make sure sent pred tags don't cross sentence boundaries
            if not sent_pred_tags[0][0] in ('O', 'B', 'U'):
                tag = sent_pred_tags[0]
                sent_pred_tags[0] = 'B'+sent_pred_tags[0][1:]
                print(f'correcting {tag} to {sent_pred_tags[0]}')

            if not sent_pred_tags[-1][0] in ('O', 'L', 'U'):
                tag = sent_pred_tags[-1]
                sent_pred_tags[-1] = 'L' + sent_pred_tags[-1][1:]
                print(f'correcting {tag} to {sent_pred_tags[-1]}')
            
            predicted_spans = bioul_tags_to_spans(sent_pred_tags)
            gold_spans = bioul_tags_to_spans(sent_tags)
            tps, fps, fns = defaultdict(int), defaultdict(int), defaultdict(int)
            for span in predicted_spans:
                if span in gold_spans:
                    tps[span[0]] += 1
                    gold_spans.remove(span)
                else:
                    fps[span[0]] += 1
            # These spans weren't predicted.
            for span in gold_spans:
                fns[span[0]] += 1
                
            tp, fp, fn = (sum(tps.values()), sum(fps.values()), sum(fns.values()))
            sent_metric_vec.append((tp, fp, fn))
            
            
            sents_tags.append(sent_tags)
            sents_pred_tags.append(sent_pred_tags)
            s = e
        return sent_metric_vec
            