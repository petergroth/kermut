import hydra

from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.distributions import MultivariateNormal
from omegaconf import DictConfig

from kermut.kernels import CompositeKernel


class KermutGP(ExactGP):
    """Gaussian Process regression model for supervised variant effects predictions.

    A specialized Gaussian Process implementation that combines sequence and structural
    information for predicting the effects of protein mutations. It extends gpytorch's
    ExactGP class and supports both composite and single kernel architectures, as well
    as zero-shot prediction capabilities through its mean function.

    Args:
        train_inputs: Training input data for the GP model. Default expects tuple of
            (one-hot sequences, sequence_embeddings, zero-shot scores).
        train_targets: Target values corresponding to the training inputs.
        likelihood: Gaussian likelihood function for the GP model.
        kernel_cfg (DictConfig): Configuration dictionary for kernel specifications,
            containing settings for sequence_kernel and structure_kernel if composite
            is True, or a single kernel configuration if composite is False.
        use_zero_shot_mean (bool, optional): Whether to use a linear mean function
            for zero-shot predictions. If True, uses LinearMean; if False, uses
            ConstantMean. Defaults to True.
        composite (bool, optional): Whether to use a composite kernel combining
            sequence and structure information. If False, uses a single kernel
            specified in kernel_cfg. Defaults to True.
        **kwargs: Additional keyword arguments passed to the kernel initialization.

    Attributes:
        covar_module: The kernel (covariance) function, either a CompositeKernel
            or a single kernel as specified by kernel_cfg.
        mean_module: The mean function, either LinearMean for zero-shot predictions
            or ConstantMean for standard GP regression.
        use_zero_shot_mean (bool): Flag indicating whether zero-shot mean function
            is being used.
    """

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
        else:
            self.mean_module = ConstantMean()

    def forward(self, x_toks, x_embed, x_zero=None) -> MultivariateNormal:
        if x_zero is None:
            x_zero = x_toks
        mean_x = self.mean_module(x_zero)
        covar_x = self.covar_module((x_toks, x_embed))
        return MultivariateNormal(mean_x, covar_x)
