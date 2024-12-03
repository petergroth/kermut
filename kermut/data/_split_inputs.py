from typing import List

import torch


def split_inputs(
    train_idx: List[bool],
    test_idx: List[bool],
    value: torch.Tensor,
):
    if value is not None:
        return value[train_idx], value[test_idx]
    else:
        return None, None
