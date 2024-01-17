import gpytorch
import hydra
import torch
from omegaconf import DictConfig
from torch.nn.functional import softplus


class ExactGPModelRBF(gpytorch.models.ExactGP):
    def __init__(
        self, train_x, train_y, likelihood, use_zero_shot: bool = False, **kwargs
    ):
        super(ExactGPModelRBF, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        # If true, will use zero-shot estimates as mean function
        self.use_zero_shot = use_zero_shot
        if use_zero_shot:
            self.register_parameter(
                "zero_shot_scale", torch.nn.Parameter(torch.tensor(1.0))
            )

    def forward(self, x):
        if self.use_zero_shot:
            # Last column is now zero-shot score
            zero_shot = x[:, -1]
            x = x[:, :-1]
            mean_x = self.mean_module(x) + self.zero_shot_scale * zero_shot
        else:
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
        use_embeddings: bool = False,
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

        self.k_1 = hydra.utils.instantiate(
            gp_cfg.model, **kermut_params, **gp_cfg.kernel_params
        )
        if "use_rbf" in gp_cfg:
            self.use_rbf = gp_cfg.use_rbf
            if gp_cfg.use_rbf:
                self.register_parameter("alpha", torch.nn.Parameter(torch.tensor(0.5)))
                self.k_2 = gpytorch.kernels.RBFKernel()
        if "use_matern" in gp_cfg:
            self.use_matern = gp_cfg.use_matern
            if gp_cfg.use_matern:
                self.register_parameter("alpha", torch.nn.Parameter(torch.tensor(0.5)))
                self.k_2 = gpytorch.kernels.MaternKernel(nu=2.5)
        self.use_embeddings = use_embeddings

    def forward(self, x):
        """
        Forward pass of GP model

        Args:
            x (torch.Tensor): If not use_embeddings and not use_zero_shot,
                x is [B, seq_len*20]. If use_embeddings, x is [B, seq_len*20 + 768].
                If use_zero_shot, x is [B, seq_len*20 + 1].
                If both use_zero_shot and use_embeddings, x is [B, seq_len*20 + 1 + 768].

        """
        if self.use_embeddings:
            x_embed = x[:, -768:]
            x = x[:, :-768]

        if self.use_zero_shot:
            # Last column is now zero-shot score
            zero_shot = x[:, -1]
            x_oh = x[:, :-1]
            mean_x = self.mean_module(x_oh) + self.zero_shot_scale * zero_shot
        else:
            x_oh = x
            mean_x = self.mean_module(x_oh)

        covar_x = self.k_1(x_oh)

        if self.use_rbf or self.use_matern:
            if self.use_embeddings:
                covar_2 = self.k_2(x_embed)
            else:
                covar_2 = self.k_2(x_oh)
            covar_x = (
                torch.sigmoid(self.alpha) * covar_x
                + (1 - torch.sigmoid(self.alpha)) * covar_2
            )

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
