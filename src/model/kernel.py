from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from gpytorch.kernels import Kernel

from src.experiments.investigate_correlations import load_protein_mpnn_outputs
from src.model.distance import KermutDistance
from src.model.utils import js_divergence, hellinger_distance, get_px1x2, Tokenizer
from src import GFP_WT, BLAT_ECOLX_WT


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

        # Compute only the lower triangular elements if x1 == x2. Include the diagonal for now.
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

    def get_params(self) -> dict:
        return {
            "p_B": self.transform_fn(self.p_B).item(),
            "p_Q": self.transform_fn(self.p_Q).item(),
            "theta": self.hellinger_fn(self.theta).item(),
            "gamma": self.hellinger_fn(self.gamma).item(),
        }


class KermutHellingerKernelMulti(Kernel):
    """Kermut-distance based Hellinger kernel with support for multiple mutations."""

    def __init__(
            self,
            conditional_prob: torch.Tensor,
            wt_sequence: torch.LongTensor,
            p_B: float = 1.0,
            p_Q: float = 1.0,
            theta: float = 1.0,
            gamma: float = 1.0,
            learnable_transform: bool = False,
            learnable_hellinger: bool = False,
    ):
        super(KermutHellingerKernelMulti, self).__init__()
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

        self.conditional_prob = conditional_prob
        self.hellinger = hellinger_distance(conditional_prob, conditional_prob)
        self.wt_sequence = wt_sequence

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs):
        # Indices where x1 and x2 differ to the WT. First column is batch, second is position.
        x1_idx = torch.argwhere(x1 != self.wt_sequence)
        x2_idx = torch.argwhere(x2 != self.wt_sequence)

        # Create all pairs of indices
        x1_expanded = x1_idx.unsqueeze(1).expand(x1_idx.size(0), x2_idx.size(0), -1)
        x2_expanded = x2_idx.unsqueeze(0).expand(x1_idx.size(0), x2_idx.size(0), -1)
        all_pairs = torch.cat((x1_expanded, x2_expanded), dim=2)
        all_pairs = all_pairs.view(-1, 4)
        # Extract batch and position indices
        batch_idx = all_pairs[:, [0, 2]]  # First column x1, second column x2
        pos_idx = all_pairs[:, [1, 3]]  # First column x1, second column x

        # Extract and transform Hellinger distances
        hn = self.hellinger[pos_idx[:, 0], pos_idx[:, 1]]
        k_hn = self.hellinger_fn(self.theta) * torch.exp(
            -self.hellinger_fn(self.gamma) * hn
        )

        # Extract conditional probabilities
        x1_toks = x1[batch_idx[:, 0], pos_idx[:, 0]]
        x2_toks = x2[batch_idx[:, 1], pos_idx[:, 1]]
        p_x1 = self.conditional_prob[pos_idx[:, 0], x1_toks]
        p_x2 = self.conditional_prob[pos_idx[:, 1], x2_toks]
        k_p_x1x2 = 1 / (
                1
                + self.transform_fn(self.p_Q)
                * torch.exp(-self.transform_fn(self.p_B) * p_x1 * p_x2)
        )

        # Sum kernel values for each sequence pair
        output = torch.zeros(x1.size(0), x2.size(0))
        # Get unique indices and original placement
        unique_indices, inverse_indices = torch.unique(
            batch_idx, return_inverse=True, dim=0
        )
        unique_indices = unique_indices.long()
        k_sum = torch.zeros(len(unique_indices))
        k_sum = torch.scatter_add(k_sum, 0, inverse_indices, k_hn * k_p_x1x2)
        output[unique_indices[:, 0], unique_indices[:, 1]] = k_sum

        return output

    def get_params(self) -> dict:
        return {
            "p_B": self.transform_fn(self.p_B).item(),
            "p_Q": self.transform_fn(self.p_Q).item(),
            "theta": self.hellinger_fn(self.theta).item(),
            "gamma": self.hellinger_fn(self.gamma).item(),
        }


if __name__ == "__main__":
    dataset = "BLAT_ECOLX"
    conditional_probs_path = Path(
        "data",
        "interim",
        dataset,
        "proteinmpnn",
        "conditional_probs_only",
        f"{dataset}.npz",
    )
    assay_path = Path("data", "processed", f"{dataset}.tsv")

    # Load data
    conditional_prob = load_protein_mpnn_outputs(conditional_probs_path, as_tensor=True)
    df = pd.read_csv(assay_path, sep="\t")

    tokenizer = Tokenizer()
    sequences = df["seq"]
    tokens = tokenizer(sequences)
    wt_sequence = tokenizer([BLAT_ECOLX_WT])[0]

    # TODO: Compute distance between two sequences
    # TODO: Batched version

    x1 = tokens[[0, 30, 100]]
    x2 = tokens[[0, 30, 100]]

    kernel = KermutHellingerKernelMulti(
        conditional_prob=conditional_prob, wt_sequence=wt_sequence
    )
    tmp = kernel(tokens[:100]).evaluate()
