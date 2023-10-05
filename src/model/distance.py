import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.utils import (
    js_divergence,
)


class KermutDistance(nn.Module):
    def __init__(self, js_exponent: float = 1.0, p_exponent: float = 1.0):
        """Kermut distance.

        The distance is composed of the following components
            - Jensen-Shannon divergence between the conditional probabilities of x and x'.
            - p(x[i]) and p(x[j]') (the probability of x and x' respectively)

        The formulation is as follows:

            d(x, x') = 1 - (1 - JS(x, x')/ln(2))^softplus(a) * (p(x[i]) * p(x[j]'))^softplus(b)

        The Jensen-Shannon divergence is divided by ln(2) to normalize it to the range [0, 1]. This quantity is then
        subtracted from 1.
        The exponents a and b are learned parameters. The default values are 1.0. The exponents are passed through the
        softplus function to ensure they are positive.


        Args:
            js_exponent (float, optional): Exponent for the Jenson-Shannon divergence. Defaults to 1.0.
            p_exponent (float, optional): Exponent for the probability. Defaults to 1.0.
        """
        super(KermutDistance, self).__init__()
        self.js_exponent = nn.Parameter(js_exponent * torch.ones(1, 1))
        self.p_exponent = nn.Parameter(p_exponent * torch.ones(1, 1))

    def forward(self, x1, x2, **kwargs):
        """Compute distance.

        Args:
            x1 (torch.Tensor): Shape (n, 20)
            x2 (torch.Tensor): Shape (n, 20)
            **kwargs: idx_1 is the amino acid index of the variants in x1 being investigated (0-19). If x1 == x2, idx_1
            is used for both x1 and x2. If x1 != x2, idx_2 is required as additional input.

        Returns:
            torch.Tensor: Shape (n, n)
        """
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

        js = js_divergence(x1, x2)
        js = torch.pow(
            (js / torch.log(torch.tensor(2.0))), F.softplus(self.js_exponent)
        )
        p_x1x2 = torch.pow(1 - p_x1x2, F.softplus(self.p_exponent))

        return js * p_x1x2

    def get_exponents(self):
        """Get exponents.

        Returns:
            dict: Dictionary of softplus applied to exponents.
        """
        return {
            "js_exponent": F.softplus(self.js_exponent.item()),
            "p_exponent": F.softplus(self.p_exponent.item()),
        }
