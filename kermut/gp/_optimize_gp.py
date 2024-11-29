from typing import Tuple

import torch

from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from omegaconf import DictConfig
from tqdm import trange


def optimize_gp(
    cfg: DictConfig,
    gp: ExactGP,
    likelihood: GaussianLikelihood,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
) -> Tuple[ExactGP, GaussianLikelihood]:
    """Train the GP model."""
    gp.train()
    likelihood.train()
    mll = ExactMarginalLogLikelihood(likelihood, gp)
    
    optimizer = torch.optim.AdamW(
        gp.parameters(),
        lr=cfg.optim.lr
    )
    
    # None inputs not allowed
    x_train = tuple([x for x in x_train if x is not None])
    
    for _ in trange(cfg.optim.n_steps, disable=not cfg.optim.progress_bar):
        optimizer.zero_grad()
        output = gp(*x_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
    return gp, likelihood