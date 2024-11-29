import hydra

from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.distributions import MultivariateNormal
from omegaconf import DictConfig

from kermut.kernels import CompositeKernel
import torch

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
            self.mean_module = LinearMean(input_size=1, bias=True)
            # self.register_parameter("zero_shot_scale", torch.nn.Parameter(torch.tensor(1.0)))
            # self.register_parameter("zero_shot_bias", torch.nn.Parameter(torch.tensor(0.0)))
        else:
            self.mean_module = ConstantMean()

    def forward(self, x_toks, x_embed, x_zero=None) -> MultivariateNormal:
        if x_zero is None:
            x_zero = x_toks
        mean_x = self.mean_module(x_zero)
        covar_x = self.covar_module((x_toks, x_embed))
        return MultivariateNormal(mean_x, covar_x)
