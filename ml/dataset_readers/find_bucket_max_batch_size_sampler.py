import logging
from typing import List, Iterable, Tuple, Optional
import random
import math

from torch.utils import data

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.samplers import BatchSampler, BucketBatchSampler

logger = logging.getLogger(__name__)


def add_noise_to_value(value: int, noise_param: float):
    noise_value = value * noise_param
    noise = random.uniform(-noise_value, noise_value)
    return value + noise


@BatchSampler.register("find_bucket_max_batch_size")
class FindBucketMaxBatchSizeBatchSampler(BucketBatchSampler):
    """
    Used to find the maximum batch size usable for the bucket sampler on a given dataset.

    We find it by taking the largest sequences and providing them as batches, starting at a size of one and increasing
    until we get a cuda error. The size just before that is the max possible batch size.


    An sampler which by default, argsorts batches with respect to the maximum input lengths `per
    batch`. You can provide a list of field names and padding keys (or pass none, in which case they
    will be inferred) which the dataset will be sorted by before doing this batching, causing inputs
    with similar length to be batched together, making computation more efficient (as less time is
    wasted on padded elements of the batch).

    # Parameters

    data_source: `data.Dataset`, required
        The pytorch `Dataset` of allennlp Instances to bucket.

        In a typical AllenNLP configuration file, this parameter does not get an entry under the
        "batch_sampler", it gets constructed separately.
    batch_size : `int`, required
        The size of each batch of instances yielded when calling the dataloader.

    sorting_keys : `List[str]`, optional
        To bucket inputs into batches, we want to group the instances by padding length, so that we
        minimize the amount of padding necessary per batch. In order to do this, we need to know
        which fields need what type of padding, and in what order.

        Specifying the right keys for this is a bit cryptic, so if this is not given we try to
        auto-detect the right keys by iterating through a few instances upfront, reading all of the
        padding keys and seeing which one has the longest length.  We use that one for padding.
        This should give reasonable results in most cases. Some cases where it might not be the
        right thing to do are when you have a `ListField[TextField]`, or when you have a really
        long, constant length `ArrayField`.

        When you need to specify this yourself, you can create an instance from your dataset and
        call `Instance.get_padding_lengths()` to see a list of all keys used in your data.  You
        should give one or more of those as the sorting keys here.

    padding_noise : `float`, optional (default=`.1`)
        When sorting by padding length, we add a bit of noise to the lengths, so that the sorting
        isn't deterministic.  This parameter determines how much noise we add, as a percentage of
        the actual padding value for each instance.

    drop_last : `bool`, (default = `False`)
        If `True`, the sampler will drop the last batch if
        its size would be less than batch_size`.

    """

    def __init__(
        self,
        data_source: data.Dataset,
        batch_size: int,
        sorting_keys: List[str] = None,
        padding_noise: float = 0.1,
        drop_last: bool = False,
    ):

        self.vocab = data_source.vocab
        self.sorting_keys = sorting_keys
        self.padding_noise = 0.0  # force real lens
        self.batch_size = batch_size
        self.data_source = data_source
        self.drop_last = drop_last

    def __iter__(self) -> Iterable[List[int]]:
        indices, lengths = self._argsort_by_padding(self.data_source)
        indices = list(reversed(indices))
        lengths = list(reversed(lengths))

        for batch_size in range(1, 1000):
            batch = indices[:batch_size]
            batch_lens = lengths[:batch_size]
            logger.info(f"Trying batch of size {batch_size} with lens {batch_lens}")
            yield batch
            logger.info(f"Batch size: {batch_size} succeeded!")
