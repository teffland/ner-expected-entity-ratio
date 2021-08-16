from typing import *

import torch


def tag_type_map(label_to_index: Dict[str, int]) -> torch.Tensor:
    """ Generate mapping matrix of tags to their types: eg B-PER -> PER. """
    T = lambda y: y.split("-")[1] if "-" in y else y
    types = list({T(y) for y in label_to_index})
    n = len(label_to_index)
    m = len(types)
    labels = [x[0] for x in sorted(list(label_to_index.items(), key=lambda x: x[1]))]
    M = torch.zeros(n, m)
    for i, l in enumerate(labels):
        l_t = T(l)
        for j, t in enumerate(types):
            if l_t == t:
                M[i, j] = 1.0

    return M
