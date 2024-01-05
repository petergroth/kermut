from pathlib import Path

import math
import numpy as np
import torch
import torch.nn as nn
from gpytorch.kernels import Kernel

from src.model.utils import hellinger_distance


class KermutP(Kernel):
    """
    Kermut-kernel version 1:
    K(x,x') = k_hn(x,x') * k_p(x,x')
    where
    k_hn(x,x') = h_scale * exp(-h_lengthscale * hn(x,x'))
    k_p(x,x') = 1 / (1 + p_Q * exp(-p_B * p(x)) * 1 / (1 + p_Q * exp(-p_B * p(x')))

    """

    def __init__(
            self,
            conditional_probs: torch.Tensor,
            wt_sequence: torch.LongTensor,
            p_B: float = 15.0,
            p_Q: float = 5.0,
            h_scale: float = 1.0,
            h_lengthscale: float = 1.0,
            learnable_transforms: bool = False,
    ):
        super(KermutP, self).__init__()

        # Register fixed parameters
        self.register_buffer("conditional_probs", conditional_probs)
        self.register_buffer("wt_sequence", wt_sequence)
        self.register_buffer(
            "hellinger", hellinger_distance(conditional_probs, conditional_probs)
        )

        self.learnable_transforms = learnable_transforms
        if learnable_transforms:
            self.register_parameter(
                name="_p_B", parameter=torch.nn.Parameter(torch.tensor(p_B))
            )
            self.register_parameter(
                name="_p_Q", parameter=torch.nn.Parameter(torch.tensor(p_Q))
            )
            self.register_parameter(
                name="_h_scale", parameter=torch.nn.Parameter(torch.tensor(h_scale))
            )
            self.register_parameter(
                name="_h_lengthscale",
                parameter=torch.nn.Parameter(torch.tensor(h_lengthscale)),
            )
            self.transform_fn = nn.Softplus()
        else:
            assert p_B > 0 and p_Q > 0 and h_scale > 0 and h_lengthscale > 0
            self.register_buffer("_p_B", torch.tensor(p_B))
            self.register_buffer("_p_Q", torch.tensor(p_Q))
            self.register_buffer("_h_scale", torch.tensor(h_scale))
            self.register_buffer("_h_lengthscale", torch.tensor(h_lengthscale))
            self.transform_fn = nn.Identity()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs):
        # Indices where x1 and x2 differ to the WT. First column is batch, second is position.
        x1_idx = torch.argwhere(x1 != self.wt_sequence)
        x2_idx = torch.argwhere(x2 != self.wt_sequence)

        # Extract and transform Hellinger distances
        hn = self.hellinger[x1_idx[:, 1].unsqueeze(1), x2_idx[:, 1].unsqueeze(0)]
        k_hn = self.h_scale * torch.exp(-self.h_lengthscale * hn)

        # Extract conditional probabilities
        x1_toks = x1[x1_idx[:, 0], x1_idx[:, 1]]
        x2_toks = x2[x2_idx[:, 0], x2_idx[:, 1]]

        p_x1 = self.conditional_probs[x1_idx[:, 1], x1_toks]
        p_x2 = self.conditional_probs[x2_idx[:, 1], x2_toks]
        # Transform probabilities
        k_p_x1 = 1 / (1 + self.p_Q * torch.exp(-self.p_B * p_x1))
        k_p_x2 = 1 / (1 + self.p_Q * torch.exp(-self.p_B * p_x2))

        # Multiply Hellinger and probability terms
        k_mult = k_hn * k_p_x1.view(-1, 1) * k_p_x2

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
            "p_B": self.p_B.item(),
            "p_Q": self.p_Q.item(),
            "h_scale": self.h_scale.item(),
            "h_lengthscale": self.h_lengthscale.item(),
        }

    @property
    def p_B(self):
        if self.learnable_transforms:
            return self.transform_fn(self._p_B)
        return self._p_B

    @property
    def p_Q(self):
        if self.learnable_transforms:
            return self.transform_fn(self._p_Q)
        return self._p_Q

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


class KermutB(Kernel):
    """
    Kermut-kernel version 2:
    K(x,x') = k_hn(x,x') * bl(x,x')
    where
    k_hn(x,x') = h_scale * exp(-h_lengthscale * hn(x,x'))
    bl(x,x') = b_scale * (bl_src * bl_tar) (BLOSUM score between sources and targets)

    """

    def __init__(
            self,
            conditional_probs: torch.Tensor,
            wt_sequence: torch.LongTensor,
            h_scale: float = 1.0,
            h_lengthscale: float = 1.0,
            b_scale: float = 1.0,
            learnable_transforms: bool = False,
    ):
        super(KermutB, self).__init__()

        # Register fixed parameters
        self.register_buffer("wt_sequence", wt_sequence)
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
                name="_h_scale", parameter=torch.nn.Parameter(torch.tensor(h_scale))
            )
            self.register_parameter(
                name="_h_lengthscale",
                parameter=torch.nn.Parameter(torch.tensor(h_lengthscale)),
            )
            self.register_parameter(
                "_b_scale", torch.nn.Parameter(torch.tensor(b_scale))
            )
            self.transform_fn = nn.Softplus()
        else:
            assert h_scale > 0 and h_lengthscale > 0 and b_scale > 0
            self.register_buffer("_h_scale", torch.tensor(h_scale))
            self.register_buffer("_h_lengthscale", torch.tensor(h_lengthscale))
            self.register_buffer("_b_scale", torch.tensor(b_scale))
            self.transform_fn = nn.Identity()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs):
        # Indices where x1 and x2 differ to the WT. First column is batch, second is position.
        x1_idx = torch.argwhere(x1 != self.wt_sequence)
        x2_idx = torch.argwhere(x2 != self.wt_sequence)

        # Extract and transform Hellinger distances
        hn = self.hellinger[x1_idx[:, 1].unsqueeze(1), x2_idx[:, 1].unsqueeze(0)]
        k_hn = self.h_scale * torch.exp(-self.h_lengthscale * hn)

        # Extract conditional probabilities
        x1_toks = x1[x1_idx[:, 0], x1_idx[:, 1]]
        x2_toks = x2[x2_idx[:, 0], x2_idx[:, 1]]

        # Compute BLOSUM scores between sources and targets.
        # E.g., k_bl(A1C, D2B) = blosum_matrix[A, D] * blosum_matrix[C, B]
        x1_wt_toks = self.wt_sequence[x1_idx[:, 1]]
        x2_wt_toks = self.wt_sequence[x2_idx[:, 1]]
        bl_src = self.blosum_matrix[x1_wt_toks][:, x2_wt_toks]
        bl_tar = self.blosum_matrix[x1_toks][:, x2_toks]
        k_bl = self.b_scale * (bl_src * bl_tar)

        # Add kernels
        k_mult = k_hn + k_bl

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
            "b_scale": self.b_scale.item(),
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
    def b_scale(self):
        return self.transform_fn(self._b_scale)


class KermutD(Kernel):
    """
    Kermut-kernel version 3:
    K(x,x') = k_hn(x,x') * k_d(x,x')
    where
    k_hn(x,x') = h_scale * exp(-h_lengthscale * hn(x,x'))
    k_d(x,x') = d_scale * exp(-d_lengthscale * d(x,x')) (distance between x and x')

    """

    def __init__(
            self,
            conditional_probs: torch.Tensor,
            wt_sequence: torch.LongTensor,
            distances: torch.Tensor,
            h_scale: float = 1.0,
            h_lengthscale: float = 1.0,
            d_scale: float = 1.0,
            d_lengthscale: float = 1.0,
            learnable_transforms: bool = False,
    ):
        super(KermutD, self).__init__()

        # Register fixed parameters
        self.register_buffer("wt_sequence", wt_sequence)
        self.register_buffer("conditional_probs", conditional_probs)
        self.register_buffer(
            "hellinger", hellinger_distance(conditional_probs, conditional_probs)
        )
        self.register_buffer("distances", distances)

        self.learnable_transforms = learnable_transforms
        if learnable_transforms:
            self.register_parameter(
                name="_h_scale", parameter=torch.nn.Parameter(torch.tensor(h_scale))
            )
            self.register_parameter(
                name="_h_lengthscale",
                parameter=torch.nn.Parameter(torch.tensor(h_lengthscale)),
            )
            self.register_parameter(
                "_d_scale", torch.nn.Parameter(torch.tensor(d_scale))
            )
            self.register_parameter(
                "_d_lengthscale", torch.nn.Parameter(torch.tensor(d_lengthscale))
            )
            self.transform_fn = nn.Softplus()
        else:
            assert (
                    h_scale > 0 and h_lengthscale > 0 and d_scale > 0 and d_lengthscale > 0
            )
            self.register_buffer("_h_scale", torch.tensor(h_scale))
            self.register_buffer("_h_lengthscale", torch.tensor(h_lengthscale))
            self.register_buffer("_d_scale", torch.tensor(d_scale))
            self.register_buffer("_d_lengthscale", torch.tensor(d_lengthscale))
            self.transform_fn = nn.Identity()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs):
        # Indices where x1 and x2 differ to the WT. First column is batch, second is position.
        x1_idx = torch.argwhere(x1 != self.wt_sequence)
        x2_idx = torch.argwhere(x2 != self.wt_sequence)

        # Extract and transform Hellinger distances
        hn = self.hellinger[x1_idx[:, 1].unsqueeze(1), x2_idx[:, 1].unsqueeze(0)]
        k_hn = self.h_scale * torch.exp(-self.h_lengthscale * hn)

        distance = self.distances[x1_idx[:, 1].unsqueeze(1), x2_idx[:, 1].unsqueeze(0)]
        k_d = self.d_scale * torch.exp(-self.d_lengthscale * distance)

        # Add kernels
        k_mult = k_hn * k_d

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
            "d_scale": self.d_scale.item(),
            "d_lengthscale": self.d_lengthscale.item(),
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
    def d_scale(self):
        if self.learnable_transforms:
            return self.transform_fn(self._d_scale)
        return self._d_scale

    @property
    def d_lengthscale(self):
        if self.learnable_transforms:
            return self.transform_fn(self._d_lengthscale)
        return self._d_lengthscale


class KermutBD(Kernel):
    """
    Kermut-kernel version 4:
    K(x,x') = k_hn(x,x') * k_d(x,x') + k_bl(x,x')
    where
    k_hn(x,x') = h_scale * exp(-h_lengthscale * hn(x,x'))
    k_d(x,x') = d_scale * exp(-d_lengthscale * d(x,x')) (distance between x and x')
    k_bl(x,x') = b_scale * (bl_src * bl_tar) (BLOSUM score between sources and targets)
    """

    def __init__(
            self,
            conditional_probs: torch.Tensor,
            wt_sequence: torch.LongTensor,
            distances: torch.Tensor,
            h_scale: float = 1.0,
            h_lengthscale: float = 1.0,
            b_scale: float = 1.0,
            d_scale: float = 1.0,
            d_lengthscale: float = 1.0,
            learnable_transforms: bool = False,
    ):
        super(KermutBD, self).__init__()

        # Register fixed parameters
        self.register_buffer("wt_sequence", wt_sequence)
        self.register_buffer("conditional_probs", conditional_probs)
        self.register_buffer(
            "hellinger", hellinger_distance(conditional_probs, conditional_probs)
        )
        self.register_buffer("distances", distances)
        blosum_matrix = np.load(Path("data", "interim", "blosum62.npy"))
        blosum_matrix = torch.tensor(blosum_matrix)
        self.register_buffer("blosum_matrix", blosum_matrix.float())

        self.learnable_transforms = learnable_transforms
        if learnable_transforms:
            self.register_parameter(
                name="_h_scale", parameter=torch.nn.Parameter(torch.tensor(h_scale))
            )
            self.register_parameter(
                name="_h_lengthscale",
                parameter=torch.nn.Parameter(torch.tensor(h_lengthscale)),
            )
            self.register_parameter(
                "_b_scale", torch.nn.Parameter(torch.tensor(b_scale))
            )
            self.register_parameter(
                "_d_scale", torch.nn.Parameter(torch.tensor(d_scale))
            )
            self.register_parameter(
                "_d_lengthscale", torch.nn.Parameter(torch.tensor(d_lengthscale))
            )
            self.transform_fn = nn.Softplus()
        else:
            assert (
                    h_scale > 0
                    and h_lengthscale > 0
                    and d_scale > 0
                    and d_lengthscale > 0
                    and b_scale > 0
            )
            self.register_buffer("_h_scale", torch.tensor(h_scale))
            self.register_buffer("_h_lengthscale", torch.tensor(h_lengthscale))
            self.register_buffer("_b_scale", torch.tensor(b_scale))
            self.register_buffer("_d_scale", torch.tensor(d_scale))
            self.register_buffer("_d_lengthscale", torch.tensor(d_lengthscale))
            self.transform_fn = nn.Identity()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs):
        # Indices where x1 and x2 differ to the WT. First column is batch, second is position.
        x1_idx = torch.argwhere(x1 != self.wt_sequence)
        x2_idx = torch.argwhere(x2 != self.wt_sequence)

        # Extract and transform Hellinger distances
        hn = self.hellinger[x1_idx[:, 1].unsqueeze(1), x2_idx[:, 1].unsqueeze(0)]
        k_hn = self.h_scale * torch.exp(-self.h_lengthscale * hn)

        # BLOSUM scores
        x1_toks = x1[x1_idx[:, 0], x1_idx[:, 1]]
        x2_toks = x2[x2_idx[:, 0], x2_idx[:, 1]]
        x1_wt_toks = self.wt_sequence[x1_idx[:, 1]]
        x2_wt_toks = self.wt_sequence[x2_idx[:, 1]]
        bl_src = self.blosum_matrix[x1_wt_toks][:, x2_wt_toks]
        bl_tar = self.blosum_matrix[x1_toks][:, x2_toks]
        k_bl = self.b_scale * (bl_src * bl_tar)

        # Distances
        distance = self.distances[x1_idx[:, 1].unsqueeze(1), x2_idx[:, 1].unsqueeze(0)]
        k_d = self.d_scale * torch.exp(-self.d_lengthscale * distance)

        # Add kernels
        k_mult = k_hn * k_d + k_bl

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
            "b_scale": self.b_scale.item(),
            "d_scale": self.d_scale.item(),
            "d_lengthscale": self.d_lengthscale.item(),
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
    def b_scale(self):
        return self.transform_fn(self._b_scale)

    @property
    def d_scale(self):
        if self.learnable_transforms:
            return self.transform_fn(self._d_scale)
        return self._d_scale

    @property
    def d_lengthscale(self):
        if self.learnable_transforms:
            return self.transform_fn(self._d_lengthscale)
        return self._d_lengthscale


class KermutBH(Kernel):
    """
    Kermut-kernel version 5:
    K(x,x') = k_hn(x,x') +( k_bl(x,x') * k_hn2(x,x'))
    where
    k_hn(x,x') = h_scale * exp(-h_lengthscale * hn(x,x'))
    k_hn2(x,x') = h_scale_2 * exp(-h_lengthscale_2 * hn(x,x'))
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
            learnable_transforms: bool = False,
    ):
        super(KermutBH, self).__init__()

        # Register fixed parameters
        self.register_buffer("wt_sequence", wt_sequence)
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
                name="_h_scale", parameter=torch.nn.Parameter(torch.tensor(h_scale))
            )
            self.register_parameter(
                name="_h_lengthscale",
                parameter=torch.nn.Parameter(torch.tensor(h_lengthscale)),
            )
            self.register_parameter(
                name="_h_scale_2", parameter=torch.nn.Parameter(torch.tensor(h_scale_2))
            )
            self.register_parameter(
                name="_h_lengthscale_2",
                parameter=torch.nn.Parameter(torch.tensor(h_lengthscale_2)),
            )
            self.transform_fn = nn.Softplus()
        else:
            assert (
                    h_scale > 0
                    and h_lengthscale > 0
                    and h_scale_2 > 0
                    and h_lengthscale_2 > 0
            )
            self.register_buffer("_h_scale", torch.tensor(h_scale))
            self.register_buffer("_h_lengthscale", torch.tensor(h_lengthscale))
            self.register_buffer("_h_scale_2", torch.tensor(h_scale_2))
            self.register_buffer("_h_lengthscale_2", torch.tensor(h_lengthscale_2))
            self.transform_fn = nn.Identity()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs):
        # Indices where x1 and x2 differ to the WT. First column is batch, second is position.
        x1_idx = torch.argwhere(x1 != self.wt_sequence)
        x2_idx = torch.argwhere(x2 != self.wt_sequence)

        # Extract and transform Hellinger distances
        hn = self.hellinger[x1_idx[:, 1].unsqueeze(1), x2_idx[:, 1].unsqueeze(0)]
        k_hn = self.h_scale * torch.exp(-self.h_lengthscale * hn)
        k_hn2 = self.h_scale_2 * torch.exp(-self.h_lengthscale_2 * hn)

        # BLOSUM scores
        x1_toks = x1[x1_idx[:, 0], x1_idx[:, 1]]
        x2_toks = x2[x2_idx[:, 0], x2_idx[:, 1]]
        x1_wt_toks = self.wt_sequence[x1_idx[:, 1]]
        x2_wt_toks = self.wt_sequence[x2_idx[:, 1]]
        bl_src = self.blosum_matrix[x1_wt_toks][:, x2_wt_toks]
        bl_tar = self.blosum_matrix[x1_toks][:, x2_toks]
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


class KermutBHNorm(Kernel):
    """
    Kermut-kernel version 6:
    K(x,x') = k_hn(x,x') +( k_bl(x,x') * k_hn2(x,x'))
    where
    k_hn(x,x') = h_scale * exp(-h_lengthscale * hn(x,x'))
    k_hn2(x,x') = h_scale_2 * exp(-h_lengthscale_2 * hn(x,x'))
    k_bl(x,x') = (bl_src * bl_tar) (BLOSUM score between sources and targets)

    This version normalizes the kernel by the total number of mutations for each entry.
    """

    def __init__(
            self,
            conditional_probs: torch.Tensor,
            wt_sequence: torch.LongTensor,
            h_scale: float = 1.0,
            h_lengthscale: float = 1.0,
            h_scale_2: float = 1.0,
            h_lengthscale_2: float = 1.0,
            learnable_transforms: bool = False,
            normalize: bool = True,
    ):
        super(KermutBHNorm, self).__init__()

        # Register fixed parameters
        self.normalize = normalize
        self.register_buffer("wt_sequence", wt_sequence)
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
                name="_h_scale", parameter=torch.nn.Parameter(torch.tensor(h_scale))
            )
            self.register_parameter(
                name="_h_lengthscale",
                parameter=torch.nn.Parameter(torch.tensor(h_lengthscale)),
            )
            self.register_parameter(
                name="_h_scale_2", parameter=torch.nn.Parameter(torch.tensor(h_scale_2))
            )
            self.register_parameter(
                name="_h_lengthscale_2",
                parameter=torch.nn.Parameter(torch.tensor(h_lengthscale_2)),
            )
            self.transform_fn = nn.Softplus()
        else:
            assert (
                    h_scale > 0
                    and h_lengthscale > 0
                    and h_scale_2 > 0
                    and h_lengthscale_2 > 0
            )
            self.register_buffer("_h_scale", torch.tensor(h_scale))
            self.register_buffer("_h_lengthscale", torch.tensor(h_lengthscale))
            self.register_buffer("_h_scale_2", torch.tensor(h_scale_2))
            self.register_buffer("_h_lengthscale_2", torch.tensor(h_lengthscale_2))
            self.transform_fn = nn.Identity()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs):
        # Indices where x1 and x2 differ to the WT. First column is batch, second is position.
        x1_idx = torch.argwhere(x1 != self.wt_sequence)
        x2_idx = torch.argwhere(x2 != self.wt_sequence)

        # Extract and transform Hellinger distances
        hn = self.hellinger[x1_idx[:, 1].unsqueeze(1), x2_idx[:, 1].unsqueeze(0)]
        k_hn = self.h_scale * torch.exp(-self.h_lengthscale * hn)
        k_hn2 = self.h_scale_2 * torch.exp(-self.h_lengthscale_2 * hn)

        # BLOSUM scores
        x1_toks = x1[x1_idx[:, 0], x1_idx[:, 1]]
        x2_toks = x2[x2_idx[:, 0], x2_idx[:, 1]]
        x1_wt_toks = self.wt_sequence[x1_idx[:, 1]]
        x2_wt_toks = self.wt_sequence[x2_idx[:, 1]]
        bl_src = self.blosum_matrix[x1_wt_toks][:, x2_wt_toks]
        bl_tar = self.blosum_matrix[x1_toks][:, x2_toks]
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

        if self.normalize:
            norm = torch.sum(one_hot_x1, dim=0).unsqueeze(1) @ torch.sum(
                one_hot_x2, dim=0
            ).unsqueeze(0)
            k_sum = k_sum / norm

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


class KermutBHMatern(Kernel):
    """
    Kermut-kernel version 7:
    K(x,x') = k_hn_mat(x,x') +( k_bl(x,x') * k_hn2_mat(x,x'))
    where
    k_hn_mat(x,x') = h_scale * constant_component * exp_component  (Matern kernel with Hellinger distance))
    k_hn2_mat(x,x') = h_scale_2 * constant_component_2 * exp_component (Matern kernel with Hellinger distance)
    k_bl(x,x') = (bl_src * bl_tar) (BLOSUM score between sources and targets)

    """

    def __init__(
            self,
            conditional_probs: torch.Tensor,
            wt_sequence: torch.LongTensor,
            nu: float = 2.5,
            h_scale: float = 1.0,
            h_lengthscale: float = 1.0,
            h_scale_2: float = 1.0,
            h_lengthscale_2: float = 1.0,
            learnable_transforms: bool = False,
    ):
        super(KermutBHMatern, self).__init__()
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        # Register fixed parameters
        self.nu = nu
        self.register_buffer("wt_sequence", wt_sequence)
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
                name="_h_scale", parameter=torch.nn.Parameter(torch.tensor(h_scale))
            )
            self.register_parameter(
                name="_h_lengthscale",
                parameter=torch.nn.Parameter(torch.tensor(h_lengthscale)),
            )
            self.register_parameter(
                name="_h_scale_2", parameter=torch.nn.Parameter(torch.tensor(h_scale_2))
            )
            self.register_parameter(
                name="_h_lengthscale_2",
                parameter=torch.nn.Parameter(torch.tensor(h_lengthscale_2)),
            )
            self.transform_fn = nn.Softplus()
        else:
            assert (
                    h_scale > 0
                    and h_lengthscale > 0
                    and h_scale_2 > 0
                    and h_lengthscale_2 > 0
            )
            self.register_buffer("_h_scale", torch.tensor(h_scale))
            self.register_buffer("_h_lengthscale", torch.tensor(h_lengthscale))
            self.register_buffer("_h_scale_2", torch.tensor(h_scale_2))
            self.register_buffer("_h_lengthscale_2", torch.tensor(h_lengthscale_2))
            self.transform_fn = nn.Identity()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs):
        # Indices where x1 and x2 differ to the WT. First column is batch, second is position.
        x1_idx = torch.argwhere(x1 != self.wt_sequence)
        x2_idx = torch.argwhere(x2 != self.wt_sequence)

        # Extract and transform Hellinger distances
        hn = self.hellinger[x1_idx[:, 1].unsqueeze(1), x2_idx[:, 1].unsqueeze(0)]
        hn_1 = hn / self.h_lengthscale
        hn_2 = hn / self.h_lengthscale_2

        if self.nu == 0.5:
            exp_component_1 = torch.exp(-hn_1)
            exp_component_2 = torch.exp(-hn_2)
            constant_component_1 = 1.0
            constant_component_2 = 1.0
        elif self.nu == 1.5:
            constant_component_1 = 1.0 + math.sqrt(3) * hn_1
            constant_component_2 = 1.0 + math.sqrt(3) * hn_2
            exp_component_1 = torch.exp(-math.sqrt(3) * hn_1)
            exp_component_2 = torch.exp(-math.sqrt(3) * hn_2)
        elif self.nu == 2.5:
            constant_component_1 = 1.0 + math.sqrt(5) * hn_1 + (5.0 / 3.0) * hn_1 ** 2
            constant_component_2 = 1.0 + math.sqrt(5) * hn_2 + (5.0 / 3.0) * hn_2 ** 2
            exp_component_1 = torch.exp(-math.sqrt(5) * hn_1)
            exp_component_2 = torch.exp(-math.sqrt(5) * hn_2)
        k_hn_1 = self.h_scale * constant_component_1 * exp_component_1
        k_hn_2 = self.h_scale_2 * constant_component_2 * exp_component_2

        # BLOSUM scores
        x1_toks = x1[x1_idx[:, 0], x1_idx[:, 1]]
        x2_toks = x2[x2_idx[:, 0], x2_idx[:, 1]]
        x1_wt_toks = self.wt_sequence[x1_idx[:, 1]]
        x2_wt_toks = self.wt_sequence[x2_idx[:, 1]]
        bl_src = self.blosum_matrix[x1_wt_toks][:, x2_wt_toks]
        bl_tar = self.blosum_matrix[x1_toks][:, x2_toks]
        k_bl = bl_src * bl_tar

        # Add kernels
        k_mult = k_hn_1 + (k_bl * k_hn_2)

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


class KermutBH_oh(Kernel):
    """
    Kermut-kernel version 8:
    K(x,x') = k_hn(x,x') +( k_bl(x,x') * k_hn2(x,x'))
    where
    k_hn(x,x') = h_scale * exp(-h_lengthscale * hn(x,x'))
    k_hn2(x,x') = h_scale_2 * exp(-h_lengthscale_2 * hn(x,x'))
    k_bl(x,x') = (bl_src * bl_tar) (BLOSUM score between sources and targets)

    This version uses one-hot encoded sequences inputs [seq_len, 20]. Functionally equivalent to KermutBH, which
    uses integer encoded sequences [seq_len, 1].
    """

    def __init__(
            self,
            conditional_probs: torch.Tensor,
            wt_sequence: torch.LongTensor,
            h_scale: float = 1.0,
            h_lengthscale: float = 1.0,
            h_scale_2: float = 1.0,
            h_lengthscale_2: float = 1.0,
            learnable_transforms: bool = False,
    ):
        super(KermutBH_oh, self).__init__()

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
            assert h_lengthscale > 0 and h_scale_2 > 0 and h_lengthscale_2 > 0
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
