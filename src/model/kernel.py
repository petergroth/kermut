from pathlib import Path

from gpytorch.kernels import Kernel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import AA_TO_IDX, COLORS
from src.experiments.investigate_correlations import load_protein_mpnn_outputs
from src.model.distance import KermutDistance
from src.model.utils import (
    get_jenson_shannon_div,
    get_euclidean_distance,
    get_probabilities,
    get_substitution_matrix,
    get_fitness_matrix,
    js_divergence,
    apply_index,
    hellinger_distance,
    get_px1x2,
)


class KermutJSKernel(Kernel):
    """Custom kernel

    NOTE: NOT VALID KERNEL.

    The kernel is the product of the following components:
    - Jensen-Shannon divergence between the conditional probabilities of x and x'
    - p(x[i]) and p(x[j]') (the probability of x and x' respectively)

    k(x, x') = (1-JS(x, x')/ln(2)) * p(x[i]) * p(x[j]')

    x and x' are probability distributions over the 20 amino acids.
    x[i] and x[j]' are the probabilities of the amino acids at position i and j respectively, where i and j are the
    indices of the amino acids in the variants being investigated.
    The Jensen-Shannon divergence term is normalized by ln(2) to ensure that the kernel is in the range [0, 1].
    Subtracting the quantity from 1 ensures that the term is 1 when the distributions are identical.
    """

    def __init__(self, js_exponent: float = 1.0, p_exponent: float = 1.0):
        super(KermutJSKernel, self).__init__()
        self.register_parameter(
            name="js_exponent",
            parameter=torch.nn.Parameter(js_exponent * torch.ones(1, 1)),
        )
        self.register_parameter(
            name="p_exponent",
            parameter=torch.nn.Parameter(p_exponent * torch.ones(1, 1)),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs):
        """Compute kernel.

        Args:
            x1 (torch.Tensor): Shape (n, 20)
            x2 (torch.Tensor): Shape (n, 20)
            **kwargs: idx_1 is the amino acid index of the variants in x1 being investigated (0-19). If x1 == x2, idx_1
            is used for both x1 and x2. If x1 != x2, idx_2 is required as additional input.

        Returns:
            torch.Tensor: Shape (n, n)
        """
        js = js_divergence(x1, x2)

        # Compute only the lower triangular elements if x1 == x2. Include the diagonal for now. TODO: Handle non-zero diagonal
        p_x1x2 = get_px1x2(x1=x1, x2=x2, **kwargs)

        # Apply transformations
        js = torch.pow((1 - js / torch.log(torch.tensor(2.0))), self.js_exponent)
        p_x1x2 = torch.pow(p_x1x2, self.p_exponent)

        return js * p_x1x2


class KermutJSD_RBFKernel(Kernel):
    """Kermut-distance based RBF kernel.

    The kernel is computed as
        k(x, x') = exp(-0.5 * d(x, x')^2 / l^2).

    The distance is computed using the Kermut distance, which is the product of the following components:
    - Jensen-Shannon divergence between the conditional probabilities of x and x'
    - p(x[i]) and p(x[j]') (the probability of x and x' respectively)

    The formulation is as follows:

            d(x, x') = 1 - (1 - JS(x, x')/ln(2))^softplus(a) * (p(x[i]) * p(x[j]'))^softplus(b)

    The Jensen-Shannon divergence is divided by ln(2) to normalize it to the range [0, 1]. This quantity is then
    subtracted from 1.
    The exponents a and b are learned parameters. The default values are 1.0. The exponents are passed through the
    softplus function to ensure they are positive.
    """

    has_lengthscale = True

    def __init__(self, **distance_params):
        super(KermutJSD_RBFKernel, self).__init__()
        self.distance_module = KermutDistance(**distance_params)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs):
        distance_matrix = self.distance_module(x1, x2, **kwargs)
        return torch.exp(-0.5 * distance_matrix.pow(2) / self.lengthscale.pow(2))


class KermutHellingerKernel(Kernel):
    # TODO: docstring
    def __init__(
            self,
            p_B: float = 1.0,
            p_Q: float = 1.0,
            theta: float = 1.0,
            gamma: float = 1.0,
            learnable_transform: bool = False,
            learnable_hellinger: bool = False,
    ):
        super(KermutHellingerKernel, self).__init__()
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
