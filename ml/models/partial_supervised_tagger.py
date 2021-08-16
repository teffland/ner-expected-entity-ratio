""" Train a crf tagger on the supervised marginal likelihood of observed tags. """
from typing import Dict, Optional, List, Any, Tuple
import warnings
import json

from overrides import overrides
import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
import numpy as np

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import PassThroughEncoder
from allennlp.modules import FeedForward, ConditionalRandomField
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import (
    CategoricalAccuracy,
    F1Measure,
    SpanBasedF1Measure,
    Average,
)
from ml.util.grammatical_transitions import allowed_transitions
from torch_struct import LinearChainCRF

INF = 1e6

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@Model.register("partial-supervised-tagger", exist_ok=True)
class PartialSupervisedTagger(Model):
    """
    The ``PartialSupervisedTagger`` encodes a sequence of text with a ``Seq2SeqEncoder``,
    then uses a Conditional Random Field model to predict a tag for each token in the sequence.

    This version allows for training the marginal likelihood of partially observed sequences,
    which are encoded with '_' tags in supervision,

        e.g., ['U', '_', '_'] instead of say ['U', 'O', 'O']

    The '_'s will be summed over when calculating the likelihood, marginalizing them out.


    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the tokens ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder that we will use in between embedding tokens and predicting output tags.
    label_namespace : ``str``, optional (default=``labels``)
        This is needed to compute the SpanBasedF1Measure metric.
        Unless you did something unusual, the default value should be what you want.
    feedforward : ``FeedForward``, optional, (default = None).
        An optional feedforward layer to apply after the encoder.
    label_encoding : ``str``, optional (default=``None``)
        Label encoding to use when calculating span f1 and constraining
        the CRF at decoding time . Valid options are "BIO", "BIOUL", "IOB1", "BMES".
        Required if ``calculate_span_f1`` or ``constrain_crf_decoding`` is true.
    constraint_type : ``str``, optional (default=``None``)
        If provided, the CRF will be constrained at decoding time
        to produce valid labels based on the specified type
        (e.g. "BIO", or "BIOUL").

        .. deprecated:: 0.6.1
           ``constraint_type`` was deprecated and replaced with
           ``label_encoding``, ``constrain_crf_decoding``, and
           ``calculate_span_f1`` in version 0.6.1. It will be removed
           in version 0.8.

    include_start_end_transitions : ``bool``, optional (default=``True``)
        Whether to include start and end transition parameters in the CRF.
    constrain_crf_decoding : ``bool``, optional (default=``None``)
        If ``True``, the CRF is constrained at decoding time to
        produce valid sequences of tags. If this is ``True``, then
        ``label_encoding`` is required. If ``None`` and
        label_encoding is specified, this is set to ``True``.
        If ``None`` and label_encoding is not specified, it defaults
        to ``False``.
    calculate_span_f1 : ``bool``, optional (default=``None``)
        Calculate span-level F1 metrics during training. If this is ``True``, then
        ``label_encoding`` is required. If ``None`` and
        label_encoding is specified, this is set to ``True``.
        If ``None`` and label_encoding is not specified, it defaults
        to ``False``.
    dropout:  ``float``, optional (detault=``None``)
    verbose_metrics : ``bool``, optional (default = False)
        If true, metrics will be returned per label class in addition
        to the overall statistics.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization loss during training.
    """

    def __init__(
        self,
        # Model
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder = None,
        feedforward: Optional[FeedForward] = None,
        dropout: Optional[float] = None,
        token_namespace: str = "tokens",
        label_namespace: str = "labels",
        label_encoding: Optional[str] = "BIOUL",
        use_transitions: bool = True,
        constrain_test_crf_decoding: bool = True,
        initializer: InitializerApplicator = InitializerApplicator(),
        # Losses
        prior_loss_weight: float = 1.0,
        entity_ratio: float = 0.2,
        entity_ratio_margin: float = 0.05,
        regularizer: Optional[RegularizerApplicator] = None,
        verbose_metrics: bool = True,
        O_cost: float = None,
    ) -> None:
        super().__init__(vocab, regularizer)

        # Encoding and tag scoring
        self.pad_idx = self.vocab.get_token_index(self.vocab._padding_token, namespace=token_namespace)
        self.token_namespace = token_namespace
        self.label_namespace = label_namespace
        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        self.hidden_dim = encoder.get_output_dim() if self.encoder else text_field_embedder.get_output_dim()
        self.feedforward = feedforward
        self.dropout = nn.Dropout(dropout) if dropout else lambda x: x
        self.scoring_dim = feedforward.get_output_dim() if feedforward else self.hidden_dim
        self.num_tags = self.vocab.get_vocab_size(label_namespace) - 1  # assume tag 0 is the 'latent' tag
        self.latent_tag_index = 0  # make sure vocab is configured to do this
        self.score_tags = TimeDistributed(Linear(self.scoring_dim, self.num_tags))

        # Crf
        self.use_transitions = use_transitions
        self.constrain_test_crf_decoding = constrain_test_crf_decoding
        self.label_encoding = label_encoding
        labels = self.vocab.get_index_to_token_vocabulary(label_namespace)
        if constrain_test_crf_decoding:
            if not label_encoding:
                raise ConfigurationError("constrain_crf_decoding is True, but " "no label_encoding was specified.")
            constraints = allowed_transitions(label_encoding, labels)[1:, 1:]  # chop off latent tag
        else:
            constraints = torch.ones(self.num_tags, self.num_tags)
        self.transition_constraints = nn.Parameter(constraints.float(), requires_grad=False)
        self.transition_params = nn.Parameter(0.001 * torch.randn_like(constraints))

        initializer(self)

        self.prior_loss_weight = prior_loss_weight
        self.entity_ratio = entity_ratio
        self.entity_ratio_margin = entity_ratio_margin

        # Metrics
        self.verbose_metrics = verbose_metrics
        self.calculate_span_f1 = label_encoding is not None
        if self.calculate_span_f1:
            if not label_encoding:
                raise ConfigurationError("calculate_span_f1 is True, but " "no label_encoding was specified.")
            self.f1_metric = SpanBasedF1Measure(vocab, tag_namespace=label_namespace, label_encoding=label_encoding)

        self.tag_metrics = {token: F1Measure(index) for index, token in labels.items()}

        self.metrics = {
            "loss/marginal_nll": Average(),
            "loss/prior_er_margin": Average(),
            "pred_entity_ratio": Average(),
        }

        self.O_cost = O_cost

    @overrides
    def forward(
        self,  # type: ignore
        tokens: Dict[str, Dict[str, torch.LongTensor]],
        tags: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        output = {"metadata": metadata}
        output.update(self.encode(tokens, **output))
        output.update(self.crf(**output))
        if tags is not None:
            output.update(self.loss(tags, tokens, **output))
            self.calc_metrics(tags, **output)
            output["tags"] = tags
        return output

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        p_tags = [
            [self.vocab.get_token_from_index(tag + 1, namespace=self.label_namespace) for tag in instance_tags]
            for instance_tags in output_dict["pred_tags"].cpu().numpy().tolist()
        ]
        for i in range(len(p_tags) - 1):
            a, b = p_tags[i], p_tags[i + 1]
            if a == "O" and not b[0] in "OBU":
                p_tags[i + 1] = "B" + b[1:]
            elif not a[0] in "OLU" and a == "O":
                p_tags[i] = "L" + a[1:]

        output_dict["pred_tags"] = p_tags
        if "tags" in output_dict:
            output_dict["tags"] = [
                [self.vocab.get_token_from_index(tag, namespace=self.label_namespace) for tag in instance_tags]
                for instance_tags in output_dict["tags"].cpu().numpy().tolist()
            ]
        output_dict.pop("pred_crf")
        output_dict.pop("constrained_pred_crf")
        output_dict.pop("encodings")
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
        for tag, metric in self.tag_metrics.items():
            f1_dict = metric.get_metric(reset=reset)
            metrics_to_return.update({f"_tags/{tag}_{k}": v for k, v in f1_dict.items()})
        if self.calculate_span_f1:
            f1_dict = self.f1_metric.get_metric(reset=reset)
            if self.verbose_metrics:
                metrics_to_return.update(f1_dict)
            else:
                metrics_to_return.update({(f"_{k}" if "overall" not in k else k): v for k, v in f1_dict.items()})
        return metrics_to_return

    def encode(self, tokens: Dict[str, Dict[str, torch.LongTensor]], **kwargs) -> Dict[str, Any]:
        output = {}
        mask = util.get_text_field_mask(tokens, padding_id=self.pad_idx)
        output["mask"] = mask

        # Run the model
        reps = self.dropout(self.text_field_embedder(tokens))
        if self.encoder:
            reps = self.dropout(self.encoder(reps))
        if self.feedforward:
            reps = self.feedforward(reps)
        output["encodings"] = reps
        return output

    def crf(self, encodings: torch.FloatTensor, mask: torch.FloatTensor, **kwargs) -> Dict[str, Any]:
        output = {}

        tag_scores = self.score_tags(encodings)  # shape: Batch size, Num tokens, Tags
        B, N, C = tag_scores.shape

        if self.O_cost is not None:
            # Subtract the O_cost from O tag potentials
            bias = torch.zeros_like(tag_scores)
            bias[:, :, 0] = self.O_cost
            tag_scores = tag_scores - bias

        output["local_potentials"] = tag_scores

        # Convert to batch size, i, c_{i+1}, c_i for torch_struct
        log_phis = self._expand_potentials(tag_scores, add_transitions=self.use_transitions)

        # Maybe constrain the potentials with grammar matrix by adding -INF to all disallowed transitions
        if self.constrain_test_crf_decoding and not self.training:
            log_phis = self._constrain_transitions(log_phis)

        lengths = mask.long().sum(-1) + 1  # torch_struct expects n+1 as the size
        # print(f"lens: {lengths}")
        output["pred_crf"] = crf = LinearChainCRF(log_phis, lengths)
        output["pred_tags"] = LinearChainCRF.struct.from_parts(crf.argmax[:, :-1])[
            0
        ]  # need to chop of last dummy node

        return output

    def loss(
        self,
        tags: torch.LongTensor,
        tokens: Dict[str, Dict[str, torch.LongTensor]],
        local_potentials: torch.FloatTensor,
        pred_crf: LinearChainCRF,
        **kwargs,
    ) -> Dict[str, Any]:
        output = {}

        constrained_pred_potentials = self._expand_potentials(
            self._constrain_potentials(tags, local_potentials),
            add_transitions=self.use_transitions,
        )

        constrained_pred_crf = LinearChainCRF(constrained_pred_potentials, lengths=pred_crf.lengths)
        output["constrained_pred_crf"] = constrained_pred_crf

        loss = self._supervised_loss(constrained_pred_crf, pred_crf)

        if self.prior_loss_weight > 0:
            eer_loss = self._prior_margin_loss(pred_crf)
            loss = loss + eer_loss

        output["loss"] = loss
        return output

    def calc_metrics(
        self,
        tags: torch.LongTensor,
        pred_crf: LinearChainCRF,
        mask: torch.BoolTensor,
        **kwargs,
    ) -> None:
        # Represent viterbi tags as "class probabilities" that we can feed into the metrics
        # and prepend with a zeros vector to fill the "latent" tags position
        tag_probs = pred_crf.argmax.sum(2)  # sum out z_{i+1} positions
        B, N, C = tag_probs.shape
        tag_probs = torch.cat([torch.zeros(B, N, 1).to(tags.device), tag_probs], dim=2)
        # print(f"tag probs: {tag_probs.shape} ... {tag_probs[0]}")
        for metric in self.tag_metrics.values():
            metric(tag_probs, tags, mask)

        if self.calculate_span_f1:
            try:
                self.f1_metric(tag_probs, tags, mask)
            except Exception as e:
                if not self.training:
                    print("******* INVALID TAGGING, SHOULDNT HAPPEN outside of training... *********\n")
                    print(e)
                    print("****************\n" * 4)
                pass

    def _supervised_loss(
        self, constrained_pred_crf: LinearChainCRF, pred_crf: LinearChainCRF, **kwargs
    ) -> Dict[str, Any]:
        """Compute the marginal likelihood of observed tags.

        This is equal to the ratio of the observed/constrained to unobserved partition functions:
          p(y_O) = sum_{y : y_O subset y} exp{phi(y)}  / sum_{y'} exp{phi(y')}
        => -logp(y_O) = logZ(y') - logZ(y_O)
        """
        loss = (pred_crf.partition - constrained_pred_crf.partition).mean()
        self.metrics["loss/marginal_nll"](loss)
        return loss

    def _prior_margin_loss(self, crf: LinearChainCRF, **kwargs) -> Dict[str, Any]:
        """Compute a margin-based marginal entity tag ratio loss on tagging posterior."""
        B, N, C, _ = crf.log_potentials.shape
        # Note: the sums over full seq lens automatically incorporate length info since crf puts zero mass on pads
        tag_marginals = crf.marginals.sum(2)  # shape: B, N, C  (sum out c_{i+1})
        E_entity_counts = tag_marginals[:, :, 1:].sum()  # =sum[tag \neq O]
        EER = E_entity_counts / tag_marginals.sum()  # / B*N
        self.metrics["pred_entity_ratio"](EER)
        dist_from_center = (EER - self.entity_ratio).abs()
        margin_loss = self.prior_loss_weight * (dist_from_center - self.entity_ratio_margin).clamp(min=0)

        self.metrics["loss/prior_er_margin"](margin_loss)
        return margin_loss

    def _expand_potentials(
        self, local_potentials: torch.FloatTensor, add_transitions: bool = True
    ) -> torch.FloatTensor:
        """ Convert per-tag potentials to linear-chain """
        B, N, C = local_potentials.shape
        potentials = local_potentials.unsqueeze(2).repeat(1, 1, C, 1)

        if add_transitions:
            transitions = self.transition_params.t()  # flip to c_{i+1}, c_i
            transitions = transitions.reshape(1, 1, C, C).repeat(B, N, 1, 1)
            potentials = potentials + transitions
        return potentials

    def _constrain_potentials(self, tags: torch.LongTensor, local_potentials: torch.FloatTensor) -> torch.FloatTensor:
        """To do partially supervised learning, we convert the provided tags into potentials for a constrained crf.

        The index 0 is assumed to mean "latent". Say we have 2 real tags and are given the sequence:
          [ 0, 1, 2, 0]

        The resulting additive observations mask would be
          [[ 0.0, 0.0 ],
           [ 0.0, -inf],
           [ -inf, 0.0],
           [ 0.0, 0.0 ]]

        If we add this to the potentials for a crf, the resulting CRF will be effectively constrained to "1","2" at 1,2
        but leaving the end positions unconstrained.
        """
        mask = torch.zeros_like(local_potentials)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                tag = tags[i, j]
                if tag > 0:
                    mask[i, j, :] = -INF
                    mask[i, j, tag - 1] = 0.0
        return local_potentials + mask

    def _constrain_transitions(self, log_phis: torch.FloatTensor, weight=INF) -> torch.FloatTensor:
        B, N, C, _ = log_phis.shape
        ok_transitions = (
            self.transition_constraints.t().reshape(1, 1, C, C).repeat(B, N, 1, 1)
        )  # flip from c_{i},c_{i+1} to c_{i+1},c_{i}
        log_phis = log_phis + -weight * (1 - ok_transitions)
        return log_phis
