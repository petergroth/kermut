from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal

from ._zero_shot_mean import ZeroShotMean
from ._composite_kernel import CompositeKernel

from omegaconf import DictConfig
from typing import Tuple, Union


from torch import Tensor

import hydra


class KermutGP(ExactGP):
    """TODO"""

    def __init__(
        self,
        train_inputs,
        train_targets,
        likelihood,
        kernel_cfg: DictConfig,
        use_zero_shot_mean: bool = True,
        composite: bool = True,
        **kwargs,
    ):
        super().__init__(train_inputs, train_targets, likelihood)
        if composite:
            self.covar_module = CompositeKernel(
                sequence_kernel=kernel_cfg.sequence_kernel,
                structure_kernel=kernel_cfg.structure_kernel,
                **kwargs,
            )
        else:
            self.covar_module = hydra.utils.instantiate(kernel_cfg.kernel, **kwargs)

        self.use_zero_shot_mean = use_zero_shot_mean
        if self.use_zero_shot_mean:
            self.mean_module = ZeroShotMean()
        else:
            self.mean_module = ConstantMean()

    def forward(self, x_toks, x_embed, x_zero) -> MultivariateNormal:
        if not self.use_zero_shot_mean:
            mean_x = self.mean_module(x_toks)  # Dummy
        else:
            mean_x = self.mean_module(x_zero)
        covar_x = self.covar_module((x_toks, x_embed))
        return MultivariateNormal(mean_x, covar_x)
