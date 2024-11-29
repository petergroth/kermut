import torch
import torch.nn as nn
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive
from typing import Optional, Dict

# Default PyTorch behaviour for distance computations
CDIST_COMPUTE_MODE = "donot_use_mm_for_euclid_dist"


class BaseKernel(Kernel):
    """Base class for all mutation kernels with common utility functions."""

    def __init__(self, wt_sequence: torch.LongTensor):
        """Initialize base kernel with wild-type sequence information."""
        super(BaseKernel, self).__init__()
        self.seq_len = wt_sequence.size(0) // 20
        wt_sequence = wt_sequence.view(self.seq_len, 20)
        self.register_buffer("wt_toks", torch.nonzero(wt_sequence)[:, 1])

    def _get_mutation_indices(self, x1: torch.Tensor, x2: torch.Tensor):
        """Extract indices where sequences differ from wild-type."""
        x1 = x1.view(-1, self.seq_len, 20)
        x2 = x2.view(-1, self.seq_len, 20)
        x1_toks = torch.nonzero(x1)[:, 2].view(x1.size(0), -1)
        x2_toks = torch.nonzero(x2)[:, 2].view(x2.size(0), -1)
        return (
            torch.argwhere(x1_toks != self.wt_toks),
            torch.argwhere(x2_toks != self.wt_toks),
            x1_toks,
            x2_toks,
        )

    def _sum_multi_mutants(
        self,
        k_mult: torch.Tensor,
        x1_idx: torch.Tensor,
        x2_idx: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        one_hot_x1 = torch.zeros(
            x1_idx[:, 0].size(0), x1_idx[:, 0].max().item() + 1
        ).to(device)
        one_hot_x2 = torch.zeros(
            x2_idx[:, 0].size(0), x2_idx[:, 0].max().item() + 1
        ).to(device)
        one_hot_x1.scatter_(1, x1_idx[:, 0].unsqueeze(1), 1)
        one_hot_x2.scatter_(1, x2_idx[:, 0].unsqueeze(1), 1)
        return torch.transpose(
            torch.transpose(k_mult @ one_hot_x2, 0, 1) @ one_hot_x1, 0, 1
        )


class SiteComparisonKernel(BaseKernel):
    def __init__(
        self,
        wt_sequence: torch.LongTensor,
        conditional_probs: torch.Tensor,
        h_lengthscale: float = 1.0,
    ):
        super(SiteComparisonKernel, self).__init__(wt_sequence)
        self.register_buffer(
            "hellinger", _hellinger_distance(conditional_probs, conditional_probs)
        )
        self.register_parameter(
            "h_lengthscale", torch.nn.Parameter(torch.tensor(h_lengthscale))
        )
        self.register_constraint("h_lengthscale", Positive())

    def forward(
        self, 
        x1_idx: torch.Tensor, 
        x2_idx: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute Hellinger distance-based kernel between mutation sites."""
        hn = self.hellinger[x1_idx[:, 1].unsqueeze(1), x2_idx[:, 1].unsqueeze(0)]
        return torch.exp(-self.h_lengthscale * hn)


class ProbabilityKernel(BaseKernel):
    """Kernel based on mutation probability differences."""

    def __init__(
        self,
        wt_sequence: torch.LongTensor,
        conditional_probs: torch.Tensor,
        p_lengthscale: float = 1.0,
        **kwargs,
    ):
        super(ProbabilityKernel, self).__init__(wt_sequence)
        self.register_buffer("conditional_probs", conditional_probs.float())
        self.register_parameter(
            "p_lengthscale", torch.nn.Parameter(torch.tensor(p_lengthscale))
        )
        self.register_constraint("p_lengthscale", Positive())

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute probability-based kernel between mutations."""
        x1_idx, x2_idx, x1_toks, x2_toks = self._get_mutation_indices(x1, x2)
        p_x1 = self.conditional_probs[x1_idx[:, 1], x1_toks[x1_idx[:, 0], x1_idx[:, 1]]]
        p_x2 = self.conditional_probs[x2_idx[:, 1], x2_toks[x2_idx[:, 0], x2_idx[:, 1]]]
        p_x1 = torch.log(p_x1)
        p_x2 = torch.log(p_x2)
        p_diff = torch.abs(p_x1.unsqueeze(1) - p_x2.unsqueeze(0))
        return torch.exp(-self.p_lengthscale * p_diff)


class DistanceKernel(BaseKernel):
    """Kernel based on spatial distances between mutation sites."""

    def __init__(
        self,
        wt_sequence: torch.LongTensor,
        coords: torch.Tensor,
        d_lengthscale: float = 1.0,
        **kwargs,
    ):
        super(DistanceKernel, self).__init__(wt_sequence)
        self.register_buffer("coords", coords.float())
        self.register_parameter(
            "d_lengthscale", torch.nn.Parameter(torch.tensor(d_lengthscale))
        )
        self.register_constraint("d_lengthscale", Positive())

    def forward(self, x1_idx: torch.Tensor, x2_idx: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute distance-based kernel between mutation sites."""
        x1_coords = self.coords[x1_idx[:, 1]]
        x2_coords = self.coords[x2_idx[:, 1]]
        distances = torch.cdist(
            x1_coords, x2_coords, p=2.0, compute_mode=CDIST_COMPUTE_MODE
        )
        return torch.exp(-self.d_lengthscale * distances)


class StructureKernel(BaseKernel):
    """Composite kernel combining Hellinger, Probability, and Distance kernels.

    Sub-kernels can be selectively included or excluded using boolean flags.
    The combination of active kernels is determined using pattern matching.
    """

    def __init__(
        self,
        wt_sequence: torch.LongTensor,
        use_site_comparison: bool = True,
        use_mutation_comparison: bool = True,
        use_distance_comparison: bool = True,
        conditional_probs: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        h_lengthscale: float = 1.0,
        d_lengthscale: float = 1.0,
        p_lengthscale: float = 1.0,
        **kwargs,
    ):
        super(StructureKernel, self).__init__(wt_sequence)
        self.use_site_comparison = use_site_comparison
        self.use_mutation_comparison = use_mutation_comparison
        self.use_distance_comparison = use_distance_comparison

        if use_site_comparison:
            assert conditional_probs is not None
            self.k_H = SiteComparisonKernel(
                wt_sequence, conditional_probs, h_lengthscale
            )

        if use_mutation_comparison:
            assert conditional_probs is not None
            self.k_p = ProbabilityKernel(wt_sequence, conditional_probs, p_lengthscale)

        if use_distance_comparison:
            assert coords is not None
            self.k_d = DistanceKernel(wt_sequence, coords, d_lengthscale)

    def forward(
        self, x1: torch.LongTensor, x2: torch.LongTensor, **kwargs
    ) -> torch.Tensor:
        # Get mutation indices
        x1_idx, x2_idx, _, _ = self._get_mutation_indices(x1, x2)

        k_mult = torch.ones(x1_idx.size(0), x2_idx.size(0), device=x1.device)
        if self.use_site_comparison:
            k_mult = k_mult * self.k_H(x1_idx, x2_idx)
        if self.use_mutation_comparison:
            k_mult = k_mult * self.k_p(x1, x2)
        if self.use_distance_comparison:
            k_mult = k_mult * self.k_d(x1_idx, x2_idx)

        return self._sum_multi_mutants(k_mult, x1_idx, x2_idx, x1.device)

    def get_params(self) -> Dict[str, float]:
        params = {}
        if self.use_site_comparison:
            params["h_lengthscale"] = self.k_H.h_lengthscale.item()
        if self.use_distance_comparison:
            params["d_lengthscale"] = self.k_d.d_lengthscale.item()
        if self.use_mutation_comparison:
            params["p_lengthscale"] = self.k_p.p_lengthscale.item()
        return params


def _hellinger_distance(p: torch.tensor, q: torch.tensor) -> torch.Tensor:
    """Compute Hellinger distance between input distributions:

    HD(p, q) = sqrt(0.5 * sum((sqrt(p) - sqrt(q))^2))

    Args:
        x1 (torch.Tensor): Shape (n, 20)
        x2 (torch.Tensor): Shape (n, 20)

    Returns:
        torch.Tensor: Shape (n, n)
    """
    batch_size = p.shape[0]
    # Compute only the lower triangular elements if p == q
    if torch.allclose(p, q):
        tril_i, tril_j = torch.tril_indices(batch_size, batch_size, offset=-1)
        hellinger_tril = torch.sqrt(
            0.5 * torch.sum((torch.sqrt(p[tril_i]) - torch.sqrt(q[tril_j])) ** 2, dim=1)
        )
        hellinger_matrix = torch.zeros((batch_size, batch_size))
        hellinger_matrix[tril_i, tril_j] = hellinger_tril
        hellinger_matrix[tril_j, tril_i] = hellinger_tril
    else:
        mesh_i, mesh_j = torch.meshgrid(
            torch.arange(batch_size), torch.arange(batch_size), indexing="ij"
        )
        mesh_i, mesh_j = mesh_i.flatten(), mesh_j.flatten()
        hellinger = torch.sqrt(
            0.5 * torch.sum((torch.sqrt(p[mesh_i]) - torch.sqrt(q[mesh_j])) ** 2, dim=1)
        )
        hellinger_matrix = hellinger.reshape(batch_size, batch_size)
    return hellinger_matrix.float()