import gpytorch
import hydra
import torch
from omegaconf import DictConfig
from torch.nn.functional import softplus


class ExactGPModelRBF(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, **kwargs):
        super(ExactGPModelRBF, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPModelLinear(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, **kwargs):
        super(ExactGPModelLinear, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.LinearKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPKermut(gpytorch.models.ExactGP):
    """
    GP class for custom kernels (with Gaussian likelihood)

    k(x, x') = k_custom(x, x')

    If specified, will add RBF-kernel and weigh by alpha:
    k(x, x') = alpha * k_custom + (1 - alpha) * k_rbf(x, x')
    where 0 <= alpha <= 1.
    """

    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        gp_cfg: DictConfig,
        use_zero_shot: bool = False,
        **kermut_params,
    ):
        super(ExactGPKermut, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        # If true, will use zero-shot estimates as mean function
        self.use_zero_shot = use_zero_shot
        if use_zero_shot:
            self.register_parameter(
                "zero_shot_scale", torch.nn.Parameter(torch.tensor(1.0))
            )

        self.covar_module = hydra.utils.instantiate(
            gp_cfg.model, **kermut_params, **gp_cfg.kernel_params
        )
        if "use_rbf" in gp_cfg:
            self.use_rbf = gp_cfg.use_rbf
            self.register_parameter("alpha", torch.nn.Parameter(torch.tensor(0.5)))
            self.rbf = gpytorch.kernels.RBFKernel()

    def forward(self, x):
        # If using zero-shot mean function, separate one-hot from zero-shot values
        if self.use_zero_shot:
            zero_shot = x[:, -1]
            x = x[:, :-1]
            # mean_x = self.mean_module(x) + softplus(self.zero_shot_scale) * zero_shot
            mean_x = self.mean_module(x) + self.zero_shot_scale * zero_shot
        else:
            mean_x = self.mean_module(x)

        covar_x = self.covar_module(x)
        if self.use_rbf:
            covar_x = torch.sigmoid(self.alpha) * covar_x + (
                1 - torch.sigmoid(self.alpha)
            ) * self.rbf(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
