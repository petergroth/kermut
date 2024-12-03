from typing import Tuple

import torch


def standardize(y_train: torch.Tensor, y_test: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = y_train.mean()
    std = y_train.std()
    y_train = (y_train - mean) / std
    y_test = (y_test - mean) / std
    return y_train, y_test
