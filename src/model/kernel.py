from pathlib import Path

import math
import numpy as np
import torch
import torch.nn as nn
from gpytorch.kernels import Kernel

from src.model.model_utils import hellinger_distance


class Kermut(Kernel):
    """
    K(x,x') = k_hn(x,x') +( k_bl(x,x') * k_hn2(x,x'))
    where
    k_hn(x,x') = h_scale * exp(-h_lengthscale * d_hn(x,x'))
    k_hn2(x,x') = h_scale_2 * exp(-h_lengthscale_2 * d_hn(x,x'))
    k_bl(x,x') = (bl_src * bl_tar) (BLOSUM score between sources and targets)
    """

    def __init__(
        self,
        conditional_probs: torch.Tensor,
        wt_sequence: torch.LongTensor,
        h_scale: float = 1.0,
        h_lengthscale: float = 1.0,
        h_scale_2: float = 1.0,
        h_lengthscale_2: float = 1.0,
        learnable_transforms: bool = True,
    ):
        super(Kermut, self).__init__()

        # Register fixed parameters
        self.seq_len = wt_sequence.size(0) // 20
        wt_sequence = wt_sequence.view(self.seq_len, 20)
        self.register_buffer("wt_toks", torch.nonzero(wt_sequence)[:, 1])
        self.register_buffer("conditional_probs", conditional_probs)
        self.register_buffer(
            "hellinger", hellinger_distance(conditional_probs, conditional_probs)
        )
        blosum_matrix = np.load(Path("data", "interim", "blosum62.npy"))
        blosum_matrix = torch.tensor(blosum_matrix)
        self.register_buffer("blosum_matrix", blosum_matrix.float())

        self.learnable_transforms = learnable_transforms
        if learnable_transforms:
            self.register_parameter(
                "_h_scale", torch.nn.Parameter(torch.tensor(h_scale))
            )
            self.register_parameter(
                "_h_lengthscale",
                torch.nn.Parameter(torch.tensor(h_lengthscale)),
            )
            self.register_parameter(
                "_h_scale_2", torch.nn.Parameter(torch.tensor(h_scale_2))
            )
            self.register_parameter(
                "_h_lengthscale_2",
                torch.nn.Parameter(torch.tensor(h_lengthscale_2)),
            )
            self.transform_fn = nn.Softplus()
        else:
            self.register_buffer("_h_scale", torch.tensor(h_scale))
            self.register_buffer("_h_lengthscale", torch.tensor(h_lengthscale))
            self.register_buffer("_h_scale_2", torch.tensor(h_scale_2))
            self.register_buffer("_h_lengthscale_2", torch.tensor(h_lengthscale_2))
            self.transform_fn = nn.Identity()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs):
        # Reshape inputs
        x1 = x1.view(-1, self.seq_len, 20)
        x2 = x2.view(-1, self.seq_len, 20)
        # 1H to tokens
        x1_toks = torch.nonzero(x1)[:, 2].view(x1.size(0), -1)
        x2_toks = torch.nonzero(x2)[:, 2].view(x2.size(0), -1)
        # Indices where x1 and x2 differ to the WT. First column is batch, second is position, third is AA.
        x1_idx = torch.argwhere(x1_toks != self.wt_toks)
        x2_idx = torch.argwhere(x2_toks != self.wt_toks)

        # Extract and transform Hellinger distances
        hn = self.hellinger[x1_idx[:, 1].unsqueeze(1), x2_idx[:, 1].unsqueeze(0)]
        k_hn = self.h_scale * torch.exp(-self.h_lengthscale * hn)
        k_hn2 = self.h_scale_2 * torch.exp(-self.h_lengthscale_2 * hn)

        # BLOSUM scores
        x1_mut_toks = x1_toks[x1_idx[:, 0], x1_idx[:, 1]]
        x2_mut_toks = x2_toks[x2_idx[:, 0], x2_idx[:, 1]]
        x1_wt_toks = self.wt_toks[x1_idx[:, 1]]
        x2_wt_toks = self.wt_toks[x2_idx[:, 1]]
        bl_src = self.blosum_matrix[x1_wt_toks][:, x2_wt_toks]
        bl_tar = self.blosum_matrix[x1_mut_toks][:, x2_mut_toks]
        k_bl = bl_src * bl_tar

        # Add kernels
        k_mult = k_hn + (k_bl * k_hn2)

        # Sum over all mutations
        one_hot_x1 = torch.zeros(x1_idx[:, 0].size(0), x1_idx[:, 0].max().item() + 1)
        one_hot_x2 = torch.zeros(x2_idx[:, 0].size(0), x2_idx[:, 0].max().item() + 1)
        one_hot_x1.scatter_(1, x1_idx[:, 0].unsqueeze(1), 1)
        one_hot_x2.scatter_(1, x2_idx[:, 0].unsqueeze(1), 1)
        k_sum = torch.transpose(
            torch.transpose(k_mult @ one_hot_x2, 0, 1) @ one_hot_x1, 0, 1
        )

        return k_sum

    def get_params(self) -> dict:
        return {
            "h_scale": self.h_scale.item(),
            "h_lengthscale": self.h_lengthscale.item(),
            "h_scale_2": self.h_scale_2.item(),
            "h_lengthscale_2": self.h_lengthscale_2.item(),
        }

    @property
    def h_scale(self):
        if self.learnable_transforms:
            return self.transform_fn(self._h_scale)
        return self._h_scale

    @property
    def h_lengthscale(self):
        if self.learnable_transforms:
            return self.transform_fn(self._h_lengthscale)
        return self._h_lengthscale

    @property
    def h_scale_2(self):
        if self.learnable_transforms:
            return self.transform_fn(self._h_scale_2)
        return self._h_scale_2

    @property
    def h_lengthscale_2(self):
        if self.learnable_transforms:
            return self.transform_fn(self._h_lengthscale_2)
        return self._h_lengthscale_2


class Kermut_no_blosum(Kernel):
    """
    K(x,x') = k_hn(x,x')
    where
    k_hn(x,x') = h_scale * exp(-h_lengthscale * d_hn(x,x'))
    """

    def __init__(
        self,
        conditional_probs: torch.Tensor,
        wt_sequence: torch.LongTensor,
        h_scale: float = 1.0,
        h_lengthscale: float = 1.0,
        learnable_transforms: bool = False,
    ):
        super(Kermut_no_blosum, self).__init__()

        # Register fixed parameters
        self.seq_len = wt_sequence.size(0) // 20
        wt_sequence = wt_sequence.view(self.seq_len, 20)
        self.register_buffer("wt_toks", torch.nonzero(wt_sequence)[:, 1])
        self.register_buffer("conditional_probs", conditional_probs)
        self.register_buffer(
            "hellinger", hellinger_distance(conditional_probs, conditional_probs)
        )
        self.learnable_transforms = learnable_transforms
        if learnable_transforms:
            self.register_parameter(
                "_h_scale", torch.nn.Parameter(torch.tensor(h_scale))
            )
            self.register_parameter(
                "_h_lengthscale",
                torch.nn.Parameter(torch.tensor(h_lengthscale)),
            )
            self.transform_fn = nn.Softplus()
        else:
            self.register_buffer("_h_scale", torch.tensor(h_scale))
            self.register_buffer("_h_lengthscale", torch.tensor(h_lengthscale))
            self.transform_fn = nn.Identity()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs):
        # Reshape inputs
        x1 = x1.view(-1, self.seq_len, 20)
        x2 = x2.view(-1, self.seq_len, 20)
        # 1H to tokens
        x1_toks = torch.nonzero(x1)[:, 2].view(x1.size(0), -1)
        x2_toks = torch.nonzero(x2)[:, 2].view(x2.size(0), -1)
        # Indices where x1 and x2 differ to the WT. First column is batch, second is position, third is AA.
        x1_idx = torch.argwhere(x1_toks != self.wt_toks)
        x2_idx = torch.argwhere(x2_toks != self.wt_toks)

        # Extract and transform Hellinger distances
        hn = self.hellinger[x1_idx[:, 1].unsqueeze(1), x2_idx[:, 1].unsqueeze(0)]
        k_hn = self.h_scale * torch.exp(-self.h_lengthscale * hn)

        # For comparative purposes
        k_mult = k_hn

        # Sum over all mutations
        one_hot_x1 = torch.zeros(x1_idx[:, 0].size(0), x1_idx[:, 0].max().item() + 1)
        one_hot_x2 = torch.zeros(x2_idx[:, 0].size(0), x2_idx[:, 0].max().item() + 1)
        one_hot_x1.scatter_(1, x1_idx[:, 0].unsqueeze(1), 1)
        one_hot_x2.scatter_(1, x2_idx[:, 0].unsqueeze(1), 1)
        k_sum = torch.transpose(
            torch.transpose(k_mult @ one_hot_x2, 0, 1) @ one_hot_x1, 0, 1
        )

        return k_sum

    def get_params(self) -> dict:
        return {
            "h_scale": self.h_scale.item(),
            "h_lengthscale": self.h_lengthscale.item(),
        }

    @property
    def h_scale(self):
        if self.learnable_transforms:
            return self.transform_fn(self._h_scale)
        return self._h_scale

    @property
    def h_lengthscale(self):
        if self.learnable_transforms:
            return self.transform_fn(self._h_lengthscale)
        return self._h_lengthscale
