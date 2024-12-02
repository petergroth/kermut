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
    """Optimizes a Gaussian Process using marginal likelihood maximization.

    Trains the GP model by minimizing the negative log marginal likelihood using 
    the AdamW optimizer. The function handles training mode activation, optimizer 
    configuration, and iterative optimization with optional progress bar display.

    Args:
        cfg: Configuration object containing optimization parameters including:
            - optim.lr: Learning rate for the AdamW optimizer
            - optim.n_steps: Number of optimization steps
            - optim.progress_bar: Boolean controlling progress bar display
        gp: The Gaussian Process model to be optimized. Must be an instance
            of ExactGP.
        likelihood: The Gaussian likelihood function associated with the GP model.
        x_train: Tuple of input tensors for training. None values in the tuple
            will be filtered out.
        y_train: Target values tensor for training.

    Returns:
        A tuple containing:
            - ExactGP: The optimized Gaussian Process model
            - GaussianLikelihood: The optimized likelihood function

    Note:
        The function uses ExactMarginalLogLikelihood as the loss function and
        optimizes model parameters using gradient descent. The progress bar
        can be toggled through the configuration.
    """
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