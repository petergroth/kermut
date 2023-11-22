import time
from pathlib import Path
from typing import Union

import pandas as pd
import torch
import torch.nn as nn
from gpytorch.kernels import Kernel

from src import GFP_WT
from src.model.utils import (
    hellinger_distance,
    get_px1x2,
    Tokenizer,
    load_conditional_probs,
)


class KermutHellingerKernel_single_mutations(Kernel):
    """Kermut-distance based Hellinger kernel."""

    def __init__(
        self,
        p_B: float = 1.0,
        p_Q: float = 1.0,
        theta: float = 1.0,
        gamma: float = 1.0,
        learnable_transform: bool = False,
        learnable_hellinger: bool = False,
    ):
        super(KermutHellingerKernel_single_mutations, self).__init__()
        if learnable_transform:
            self.register_parameter(
                name="p_B", parameter=torch.nn.Parameter(torch.tensor(p_B))
            )
            self.register_parameter(
                name="p_Q", parameter=torch.nn.Parameter(torch.tensor(p_Q))
            )
            self.transform_fn = nn.Softplus()
        else:
            assert p_B > 0 and p_Q > 0
            self.p_B = torch.tensor(p_B)
            self.p_Q = torch.tensor(p_Q)
            self.transform_fn = nn.Identity()

        if learnable_hellinger:
            self.register_parameter(
                name="theta", parameter=torch.nn.Parameter(torch.tensor(theta))
            )
            self.register_parameter(
                name="gamma", parameter=torch.nn.Parameter(torch.tensor(gamma))
            )
            self.hellinger_fn = nn.Softplus()
        else:
            assert theta > 0 and gamma > 0
            self.theta = torch.tensor(theta)
            self.gamma = torch.tensor(gamma)
            self.hellinger_fn = nn.Identity()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs):
        """Compute kernel.

        Kernel is computed as the product of a Hellinger kernel and a probability kernel:
            k(x, x') = k_h(x, x') * k_p(x, x')

        The Hellinger kernel is computed as:
            k_h(x, x') = theta * exp(-gamma * d_h(x, x'))
        where theta, gamme are positive values and d_h is the Hellinger distance.

        The probability kernel is computed as:
            k_p(x, x') = 1 / (1 + B * exp(-Q * p(x[i]) * p(x[j]')))
        where B, Q are positive values and p(x[i]), p(x[j]') are the probabilities of the amino acids at position i
        and j.

        Args:
            x1 (torch.Tensor): Shape (n, 20)
            x2 (torch.Tensor): Shape (n, 20)
            **kwargs: idx_1 is the amino acid index of the variants in x1 being investigated (0-19). If x1 == x2, idx_1
            is used for both x1 and x2. If x1 != x2, idx_2 is required as additional input.

        Returns:
            torch.Tensor: Shape (n, n)
        """

        hd = hellinger_distance(x1, x2)
        k_hd = self.hellinger_fn(self.theta) * torch.exp(
            -self.hellinger_fn(self.gamma) * hd
        )
        p_x1x2 = get_px1x2(x1=x1, x2=x2, **kwargs)

        k_p = 1 / (
            1
            + self.transform_fn(self.p_Q)
            * torch.exp(-self.transform_fn(self.p_B) * p_x1x2)
        )

        return k_hd * k_p

    def get_params(self) -> dict:
        return {
            "p_B": self.transform_fn(self.p_B).item(),
            "p_Q": self.transform_fn(self.p_Q).item(),
            "theta": self.hellinger_fn(self.theta).item(),
            "gamma": self.hellinger_fn(self.gamma).item(),
        }


class KermutHellingerKernelMulti_old(Kernel):
    """Kermut-distance based Hellinger kernel with support for multiple mutations."""

    def __init__(
        self,
        conditional_probs: torch.Tensor,
        wt_sequence: torch.LongTensor,
        p_B: float = 15.0,
        p_Q: float = 5.0,
        theta: float = 1.0,
        gamma: float = 1.0,
        learnable_transform: bool = False,
        learnable_hellinger: bool = False,
    ):
        super(KermutHellingerKernelMulti_old, self).__init__()
        self.learnable_transform = learnable_transform
        self.learnable_hellinger = learnable_hellinger

        # If learnable, pass parameters through softplus function to ensure positivity during learning
        if learnable_transform:
            self.register_parameter(
                name="_p_B", parameter=torch.nn.Parameter(torch.tensor(p_B))
            )
            self.register_parameter(
                name="_p_Q", parameter=torch.nn.Parameter(torch.tensor(p_Q))
            )
            self.transform_fn = nn.Softplus()
        else:
            assert p_B > 0 and p_Q > 0
            self.register_buffer("_p_B", torch.tensor(p_B))
            self.register_buffer("_p_Q", torch.tensor(p_Q))
            self.transform_fn = nn.Identity()

        if learnable_hellinger:
            self.register_parameter(
                name="_theta", parameter=torch.nn.Parameter(torch.tensor(theta))
            )
            self.register_parameter(
                name="_gamma", parameter=torch.nn.Parameter(torch.tensor(gamma))
            )
            self.hellinger_fn = nn.Softplus()
        else:
            assert theta > 0 and gamma > 0
            self.register_buffer("_theta", torch.tensor(theta))
            self.register_buffer("_gamma", torch.tensor(gamma))
            self.hellinger_fn = nn.Identity()

        assert len(conditional_probs) == len(wt_sequence)
        self.register_buffer("conditional_probs", conditional_probs)
        self.register_buffer(
            "hellinger", hellinger_distance(conditional_probs, conditional_probs)
        )
        self.register_buffer("wt_sequence", wt_sequence)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs):
        assert x1.shape[1] == x2.shape[1]
        assert x1.shape[1] == self.wt_sequence.shape[0]

        # Indices where x1 and x2 differ to the WT. First column is batch, second is position.
        x1_idx = torch.argwhere(x1 != self.wt_sequence)
        x2_idx = torch.argwhere(x2 != self.wt_sequence)

        # Get batch and position indices of all mutation pairs (> N*N)
        batch_idx_x1 = x1_idx[:, 0].repeat_interleave(x2_idx[:, 0].shape[0])
        batch_idx_x2 = x2_idx[:, 0].repeat(x1_idx[:, 0].shape[0])
        pos_idx_x1 = x1_idx[:, 1].repeat_interleave(x2_idx[:, 1].shape[0])
        pos_idx_x2 = x2_idx[:, 1].repeat(x1_idx[:, 1].shape[0])

        # Extract and transform Hellinger distances
        hn = self.hellinger[pos_idx_x1, pos_idx_x2]
        k_hn = self.theta * torch.exp(-self.gamma * hn)

        # Extract conditional probabilities
        x1_toks = x1[batch_idx_x1, pos_idx_x1]
        x2_toks = x2[batch_idx_x2, pos_idx_x2]
        p_x1 = self.conditional_probs[pos_idx_x1, x1_toks]
        p_x2 = self.conditional_probs[pos_idx_x2, x2_toks]
        k_p_x1 = 1 / (1 + self.p_Q * torch.exp(-self.p_B * p_x1))
        k_p_x2 = 1 / (1 + self.p_Q * torch.exp(-self.p_B * p_x2))
        k_p_x1x2 = k_p_x1 * k_p_x2

        # Get unique indices and original placement
        unique_indices, inverse_indices = torch.unique(
            torch.cat((batch_idx_x1.unsqueeze(1), batch_idx_x2.unsqueeze(1)), -1),
            return_inverse=True,
            dim=0,
        )

        k_sum = torch.zeros(len(unique_indices))
        k_sum = torch.scatter_add(k_sum, 0, inverse_indices, k_hn * k_p_x1x2)
        output = torch.zeros(x1.size(0), x2.size(0))
        output[unique_indices[:, 0], unique_indices[:, 1]] = k_sum
        return output

    def get_params(self) -> dict:
        return {
            "p_B": self.p_B.item(),
            "p_Q": self.p_Q.item(),
            "theta": self.theta.item(),
            "gamma": self.gamma.item(),
        }

    @property
    def p_B(self):
        if self.learnable_transform:
            return self.transform_fn(self._p_B)
        return self._p_B

    @property
    def p_Q(self):
        if self.learnable_transform:
            return self.transform_fn(self._p_Q)
        return self._p_Q

    @property
    def theta(self):
        if self.learnable_hellinger:
            return self.hellinger_fn(self._theta)
        return self._theta

    @property
    def gamma(self):
        if self.learnable_hellinger:
            return self.hellinger_fn(self._gamma)
        return self._gamma


class KermutHellingerKernel(Kernel):
    """Kermut-distance based Hellinger kernel with support for multiple mutations."""

    def __init__(
        self,
        conditional_probs: torch.Tensor,
        wt_sequence: torch.LongTensor,
        p_B: float = 15.0,
        p_Q: float = 5.0,
        theta: float = 1.0,
        gamma: float = 1.0,
        learnable_transform: bool = False,
        learnable_hellinger: bool = False,
            blosum: bool = False,
            use_distance: bool = False,
            distances: Union[torch.Tensor, None] = None,
    ):
        super(KermutHellingerKernel, self).__init__()
        self.learnable_transform = learnable_transform
        self.learnable_hellinger = learnable_hellinger

        ################
        # Pi * Pj
        ################
        if learnable_transform:
            self.register_parameter(
                name="_p_B", parameter=torch.nn.Parameter(torch.tensor(p_B))
            )
            self.register_parameter(
                name="_p_Q", parameter=torch.nn.Parameter(torch.tensor(p_Q))
            )
            self.transform_fn = nn.Softplus()
        else:
            assert p_B > 0 and p_Q > 0
            self.register_buffer("_p_B", torch.tensor(p_B))
            self.register_buffer("_p_Q", torch.tensor(p_Q))
            self.transform_fn = nn.Identity()

        ################
        # Hellinger
        ################
        if learnable_hellinger:
            self.register_parameter(
                name="_theta", parameter=torch.nn.Parameter(torch.tensor(theta))
            )
            self.register_parameter(
                name="_gamma", parameter=torch.nn.Parameter(torch.tensor(gamma))
            )
            self.hellinger_fn = nn.Softplus()
        else:
            assert theta > 0 and gamma > 0
            self.register_buffer("_theta", torch.tensor(theta))
            self.register_buffer("_gamma", torch.tensor(gamma))
            self.hellinger_fn = nn.Identity()

        ################
        # Distance
        ################
        self.use_distance = use_distance
        if use_distance:
            assert distances is not None
            self.register_buffer("distances", distances)
            self.register_parameter("_phi", torch.nn.Parameter(torch.tensor(1.0)))
            self.register_parameter("_psi", torch.nn.Parameter(torch.tensor(1.0)))
            self.distance_fn = nn.Softplus()

        ################
        # BLOSUM
        ################
        self.blosum = blosum
        if self.blosum:
            blosum_matrix = torch.load(Path("data", "interim", "blosum62.pt"))
            self.register_buffer("blosum_matrix", blosum_matrix.float())
            self.register_parameter(
                "_blosum_scale", torch.nn.Parameter(torch.tensor(1.0))
            )
            self.blosum_fn = nn.Softplus()

        ################
        # Remaining
        ################
        self.register_buffer("conditional_probs", conditional_probs)
        self.register_buffer(
            "hellinger", hellinger_distance(conditional_probs, conditional_probs)
        )
        self.register_buffer("wt_sequence", wt_sequence)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs):
        # Indices where x1 and x2 differ to the WT. First column is batch, second is position.
        x1_idx = torch.argwhere(x1 != self.wt_sequence)
        x2_idx = torch.argwhere(x2 != self.wt_sequence)

        # Extract and transform Hellinger distances
        hn = self.hellinger[x1_idx[:, 1].unsqueeze(1), x2_idx[:, 1].unsqueeze(0)]
        k_hn = self.theta * torch.exp(-self.gamma * hn)

        # Extract conditional probabilities
        x1_toks = x1[x1_idx[:, 0], x1_idx[:, 1]]
        x2_toks = x2[x2_idx[:, 0], x2_idx[:, 1]]

        # Add BLOSUM component
        if self.blosum:
            # Compute BLOSUM scores between sources and targets.
            # E.g., k_bl(A1C, D2B) = blosum_matrix[A, D] * blosum_matrix[C, B]
            x1_wt_toks = self.wt_sequence[x1_idx[:, 1]]
            x2_wt_toks = self.wt_sequence[x2_idx[:, 1]]
            bl_src = self.blosum_matrix[x1_wt_toks][:, x2_wt_toks]
            bl_tar = self.blosum_matrix[x1_toks][:, x2_toks]
            k_bl = bl_src * bl_tar
            k_mult = k_hn + self.blosum_scale * k_bl

        else:
            p_x1 = self.conditional_probs[x1_idx[:, 1], x1_toks]
            p_x2 = self.conditional_probs[x2_idx[:, 1], x2_toks]
            # Transform probabilities
            k_p_x1 = 1 / (1 + self.p_Q * torch.exp(-self.p_B * p_x1))
            k_p_x2 = 1 / (1 + self.p_Q * torch.exp(-self.p_B * p_x2))

            # Multiply Hellinger and probability terms
            k_mult = k_hn * k_p_x1.view(-1, 1) * k_p_x2

        if self.use_distance:
            distance = self.distances[
                x1_idx[:, 1].unsqueeze(1), x2_idx[:, 1].unsqueeze(0)
            ]
            k_mult += self.phi * torch.exp(-self.psi * distance)

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
        if self.blosum:
            return {
                "theta": self.theta.item(),
                "gamma": self.gamma.item(),
                "blosum_scale": self.blosum_scale.item(),
            }
        return {
            "p_B": self.p_B.item(),
            "p_Q": self.p_Q.item(),
            "theta": self.theta.item(),
            "gamma": self.gamma.item(),
        }

    @property
    def p_B(self):
        if self.learnable_transform:
            return self.transform_fn(self._p_B)
        return self._p_B

    @property
    def p_Q(self):
        if self.learnable_transform:
            return self.transform_fn(self._p_Q)
        return self._p_Q

    @property
    def theta(self):
        if self.learnable_hellinger:
            return self.hellinger_fn(self._theta)
        return self._theta

    @property
    def gamma(self):
        if self.learnable_hellinger:
            return self.hellinger_fn(self._gamma)
        return self._gamma

    @property
    def blosum_scale(self):
        return self.blosum_fn(self._blosum_scale)

    @property
    def phi(self):
        return self.distance_fn(self._phi)

    @property
    def psi(self):
        return self.distance_fn(self._psi)


class KermutHellingerKernelSequential(Kernel):
    """Kermut-distance based Hellinger kernel with support for multiple mutations."""

    def __init__(
        self,
        conditional_probs: torch.Tensor,
        wt_sequence: torch.LongTensor,
        p_B: float = 15.0,
        p_Q: float = 5.0,
        theta: float = 1.0,
        gamma: float = 1.0,
        learnable_transform: bool = False,
        learnable_hellinger: bool = False,
    ):
        super(KermutHellingerKernelSequential, self).__init__()
        self.learnable_transform = learnable_transform
        self.learnable_hellinger = learnable_hellinger

        # If learnable, pass parameters through softplus function to ensure positivity during learning
        if learnable_transform:
            self.register_parameter(
                name="_p_B", parameter=torch.nn.Parameter(torch.tensor(p_B))
            )
            self.register_parameter(
                name="_p_Q", parameter=torch.nn.Parameter(torch.tensor(p_Q))
            )
            self.transform_fn = nn.Softplus()
        else:
            assert p_B > 0 and p_Q > 0
            self.register_buffer("_p_B", torch.tensor(p_B))
            self.register_buffer("_p_Q", torch.tensor(p_Q))
            self.transform_fn = nn.Identity()

        if learnable_hellinger:
            self.register_parameter(
                name="_theta", parameter=torch.nn.Parameter(torch.tensor(theta))
            )
            self.register_parameter(
                name="_gamma", parameter=torch.nn.Parameter(torch.tensor(gamma))
            )
            self.hellinger_fn = nn.Softplus()
        else:
            assert theta > 0 and gamma > 0
            self.register_buffer("_theta", torch.tensor(theta))
            self.register_buffer("_gamma", torch.tensor(gamma))
            self.hellinger_fn = nn.Identity()

        self.register_buffer("conditional_probs", conditional_probs)
        self.register_buffer(
            "hellinger", hellinger_distance(conditional_probs, conditional_probs)
        )
        self.register_buffer("wt_sequence", wt_sequence)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs):
        # Indices where x1 and x2 differ to the WT. First column is batch, second is position.
        output = torch.zeros(x1.size(0), x2.size(0))
        for i in range(x1.shape[0]):
            x1_idx = torch.argwhere(x1[i] != self.wt_sequence)[:, 0]
            p_x1 = self.conditional_probs[x1_idx, x1[i][x1_idx]]
            for j in range(x2.shape[0]):
                x2_idx = torch.argwhere(x2[j] != self.wt_sequence)[:, 0]
                # Hellinger contribution
                hn = self.hellinger[
                    x1_idx.repeat_interleave(x2_idx.shape[0]),
                    x2_idx.repeat(x1_idx.shape[0]),
                ]
                k_hn = self.theta * torch.exp(-self.gamma * hn)
                # Probability contribution
                p_x2 = self.conditional_probs[x2_idx, x2[j][x2_idx]]

                # All possible combinations of x1 and x2
                k_p_x1 = 1 / (1 + self.p_Q * torch.exp(-self.p_B * p_x1))
                k_p_x2 = 1 / (1 + self.p_Q * torch.exp(-self.p_B * p_x2))
                k_p_x1x2 = torch.outer(k_p_x1, k_p_x2)

                output[i, j] = torch.sum(k_hn * k_p_x1x2.flatten())
        return output

    def get_params(self) -> dict:
        return {
            "p_B": self.p_B.item(),
            "p_Q": self.p_Q.item(),
            "theta": self.theta.item(),
            "gamma": self.gamma.item(),
        }

    @property
    def p_B(self):
        if self.learnable_transform:
            return self.transform_fn(self._p_B)
        return self._p_B

    @property
    def p_Q(self):
        if self.learnable_transform:
            return self.transform_fn(self._p_Q)
        return self._p_Q

    @property
    def theta(self):
        if self.learnable_hellinger:
            return self.hellinger_fn(self._theta)
        return self._theta

    @property
    def gamma(self):
        if self.learnable_hellinger:
            return self.hellinger_fn(self._gamma)
        return self._gamma


class KermutP(Kernel):
    """Kermut-distance based Hellinger kernel with support for multiple mutations."""

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
    """Kermut-distance based Hellinger kernel with support for multiple mutations."""

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
        blosum_matrix = torch.load(Path("data", "interim", "blosum62.pt"))
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
        # TODO: Is k_bl a kernel?

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
    """Kermut-distance based Hellinger kernel with support for multiple mutations."""

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
    """Kermut-distance based Hellinger kernel with support for multiple mutations."""

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
        blosum_matrix = torch.load(Path("data", "interim", "blosum62.pt"))
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


def _forward_pass():
    dataset = "GFP"
    conditional_probs_method = "ProteinMPNN"
    # conditional_probs_method = "esm2"
    assay_path = Path("data", "processed", f"{dataset}.tsv")

    # Load data
    conditional_probs = load_conditional_probs(dataset, conditional_probs_method)
    df = pd.read_csv(assay_path, sep="\t")
    df = df[df["n_muts"] <= 2]
    df = df.iloc[:2000]
    y = df["delta_fitness"].values
    y = torch.tensor(y)

    tokenizer = Tokenizer()
    sequences = df["seq"]
    tokens = tokenizer(sequences)
    wt_sequence = tokenizer([GFP_WT])[0]
    assert len(conditional_probs) == len(wt_sequence)

    model_kwargs = {
        "conditional_probs": conditional_probs,
        "wt_sequence": wt_sequence,
        "p_B": 15.0,
        "p_Q": 5.0,
        "theta": 1.0,
        "gamma": 1.0,
        "learnable_transform": False,
        "learnable_hellinger": False,
        "blosum": True,  # NOTE
    }

    kernel = KermutHellingerKernel(**model_kwargs)

    t0 = time.time()
    out = kernel(tokens)  # IF PROFILING, INDENT
    out = out.evaluate()
    t1 = time.time()
    print(f"Elapsed time: {t1 - t0}")


if __name__ == "__main__":
    _forward_pass()
