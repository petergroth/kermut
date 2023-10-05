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
)


class KermutKernel(Kernel):
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
        super(KermutKernel, self).__init__()
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
        batch_size = x1.shape[0]

        # Compute only the lower triangular elements if x1 == x2. Include the diagonal for now. TODO: Handle non-zero diagonal
        if torch.allclose(x1, x2):
            tril_i, tril_j = torch.tril_indices(batch_size, batch_size, offset=0)
            p_x1_tril = x1[tril_i]
            p_x2_tril = x2[tril_j]
            p_x1 = p_x1_tril[torch.arange(tril_i.numel()), kwargs["idx_1"][tril_i]]
            p_x2 = p_x2_tril[torch.arange(tril_j.numel()), kwargs["idx_1"][tril_j]]
            # Build full matrix
            p_x1x2 = torch.zeros((batch_size, batch_size))
            p_x1x2[tril_i, tril_j] = p_x1 * p_x2
            p_x1x2[tril_j, tril_i] = p_x1 * p_x2
        else:
            mesh_i, mesh_j = torch.meshgrid(
                torch.arange(batch_size), torch.arange(batch_size), indexing="ij"
            )
            mesh_i, mesh_j = mesh_i.flatten(), mesh_j.flatten()
            p_x1 = x1[mesh_i][torch.arange(mesh_i.numel()), kwargs["idx_1"][mesh_i]]
            p_x2 = x2[mesh_j][torch.arange(mesh_j.numel()), kwargs["idx_2"][mesh_j]]
            p_x1x2 = p_x1 * p_x2
            p_x1x2 = p_x1x2.reshape(batch_size, batch_size)

        # Apply transformations
        js = torch.pow((1 - js / torch.log(torch.tensor(2.0))), self.js_exponent)
        p_x1x2 = torch.pow(p_x1x2, self.p_exponent)

        return js * p_x1x2


class KermutRBFKernel(Kernel):
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
        super(KermutRBFKernel, self).__init__()
        self.distance_module = KermutDistance(**distance_params)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs):
        distance_matrix = self.distance_module(x1, x2, **kwargs)
        return torch.exp(-0.5 * distance_matrix.pow(2) / self.lengthscale.pow(2))
