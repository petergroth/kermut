from typing import Tuple

import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from tqdm import trange


def optimize_gp(
    gp: ExactGP,
    likelihood: GaussianLikelihood,
    train_inputs: Tuple[torch.Tensor, ...],
    train_targets: torch.Tensor,
    lr: float = 3.0e-4,
    n_steps: int = 150,
    progress_bar: bool = True,
) -> Tuple[ExactGP, GaussianLikelihood]:
    """Optimizes a Gaussian Process using marginal likelihood maximization.

    Trains the GP model by minimizing the negative log marginal likelihood using
    the AdamW optimizer. The function handles training mode activation, optimizer
    configuration, and iterative optimization with optional progress bar display.

    Args:
        gp: The Gaussian Process model to be optimized. Must be an instance
            of ExactGP.
        likelihood: The Gaussian likelihood function associated with the GP model.
        train_inputs: Tuple of input tensors for training. None values in the tuple
            will be filtered out.
        train_targets: Target values tensor for training.
        lr: Learning rate for the AdamW optimizer. Default is 3.0e-4.
        n_steps: Number of optimization steps. Default is 150.
        progress_bar: Boolean controlling progress bar display. Default is True.

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

    optimizer = torch.optim.AdamW(gp.parameters(), lr=lr)

    # None inputs not allowed
    x_train = tuple([x for x in train_inputs if x is not None])
    y_train = train_targets

    for _ in trange(n_steps, disable=not progress_bar):
        optimizer.zero_grad()
        output = gp(*x_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
    return gp, likelihood
