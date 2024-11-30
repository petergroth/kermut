from typing import Dict, Tuple

from gpytorch.likelihoods import GaussianLikelihood
from omegaconf import DictConfig
from gpytorch.priors import HalfCauchyPrior
import torch

from ._gp import KermutGP

    
def instantiate_gp(
    cfg: DictConfig, 
    train_inputs: Tuple[torch.Tensor, ...], 
    train_targets: torch.Tensor,
    gp_inputs: Dict,
    ) -> Tuple[KermutGP, GaussianLikelihood]:
    if cfg.kernel.use_prior:
        noise_prior = HalfCauchyPrior(
            scale=cfg.kernel.noise_prior_scale
        )
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
        **gp_inputs
    )
    if cfg.use_gpu and torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
        
    return gp, likelihood