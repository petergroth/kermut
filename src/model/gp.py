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

    k(x, x') = k_kermut(x, x')

    If specified, will add RBF-kernel and weigh by alpha:
    k(x, x') = alpha * k_kermut() + (1 - alpha) * k_rbf()
    where 0 <= alpha <= 1.

    If specified, will use (precomputed) zero-shot estimates as mean function:
    mean(x) = intercept + scale * zero_shot_score(x)

    Note: GPyTorch requires kernels to operate on a single tensor x, which for kermut is flattened one-hot encoded
    sequences. If using zero-shot estimates for mean function, the zero-shot score is concatenated at the end.
    If furthermore using pre-computed embeddings, these are concatenated at the end as well.

    Example:
    - Kermut kernel only: x is [B, seq_len*20]
    - Kermut kernel + zero-shot estimates: x is [B, seq_len*20 + 1]
    - Kermut kernel + zero-shot estimates + embeddings: x is [B, seq_len*20 + 1 + 768] (in the case of MSA Transformer embeddings)
    - Kermut kernel + embeddings: x is [B, seq_len*20 + 768] (in the case of MSA Transformer embeddings)

    """

    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        gp_cfg: DictConfig,
        use_zero_shot: bool = True,
        use_embeddings: bool = True,
        embedding_dim: int = 768,
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
        if use_embeddings:
            self.embedding_dim = embedding_dim

    def forward(self, x):
        """
        Forward pass of GP model

        Args:
            x (torch.Tensor): See class docstring for details on dimensions

        """

        # Unpack input tensor
        if self.use_embeddings:
            x_embed = x[:, -self.embedding_dim :]
            x = x[:, : -self.embedding_dim]

        if self.use_zero_shot:
            zero_shot = x[:, -1]
            x_oh = x[:, :-1]
            mean_x = self.mean_module(x_oh) + self.zero_shot_scale * zero_shot
        else:
            x_oh = x
            mean_x = self.mean_module(x_oh)

        # Kermut kernel
        covar_x = self.k_1(x_oh)

        if self.use_rbf or self.use_matern:
            if self.use_embeddings:
                covar_2 = self.k_2(x_embed)
            else:
                covar_2 = self.k_2(x_oh)

            # Weighted sum of kernels
            covar_x = (
                torch.sigmoid(self.alpha) * covar_x
                + (1 - torch.sigmoid(self.alpha)) * covar_2
            )

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
