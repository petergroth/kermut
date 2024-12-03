from typing import Dict, Tuple

import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import HalfCauchyPrior
from omegaconf import DictConfig

from ._gp import KermutGP


def instantiate_gp(
    cfg: DictConfig,
    train_inputs: Tuple[torch.Tensor, ...],
    train_targets: torch.Tensor,
    gp_inputs: Dict,
) -> Tuple[KermutGP, GaussianLikelihood]:
    """Instantiates a KermutGP model and its associated Gaussian likelihood.

    Args:
        cfg: Configuration object containing model parameters including:
            - kernel.use_prior: Boolean indicating whether to use prior
            - kernel.noise_prior_scale: Scale parameter for HalfCauchy prior
            - kernel.use_structure_kernel: Boolean for structure kernel usage
            - kernel.use_sequence_kernel: Boolean for sequence kernel usage
            - kernel.use_zero_shot: Boolean for zero-shot mean
            - use_gpu: Boolean indicating GPU usage preference
        train_inputs: Tuple of torch tensors containing training input features.
            None values in the tuple will be filtered out.
        train_targets: Torch tensor containing training target values.
        gp_inputs: Dictionary containing additional inputs for the KermutGP model.

    Returns:
        A tuple containing:
            - KermutGP: The instantiated Gaussian Process model
            - GaussianLikelihood: The associated likelihood function

    Note:
        The function will automatically move the model and likelihood to GPU if
        cfg.use_gpu is True and a CUDA device is available.
    """

    if cfg.kernel.use_prior:
        noise_prior = HalfCauchyPrior(scale=cfg.kernel.noise_prior_scale)
    else:
        noise_prior = None

    likelihood = GaussianLikelihood(noise_prior=noise_prior)
    composite = cfg.kernel.use_structure_kernel and cfg.kernel.use_sequence_kernel
    train_inputs = tuple([x for x in train_inputs if x is not None])

    gp = KermutGP(
        train_inputs,
        train_targets,
        likelihood,
        kernel_cfg=cfg.kernel,
        use_zero_shot_mean=cfg.kernel.use_zero_shot,
        composite=composite,
        **gp_inputs,
    )
    if cfg.use_gpu and torch.cuda.is_available():
        gp = gp.cuda()
        likelihood = likelihood.cuda()

    return gp, likelihood
