import gpytorch
import hydra
import torch
from omegaconf import DictConfig

from src.model.kernel import KermutHellingerKernel


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


class ExactGPModelKermut(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, **kermut_params):
        super(ExactGPModelKermut, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = KermutHellingerKernel(**kermut_params)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPKermut(gpytorch.models.ExactGP):
    def __init__(
            self, train_x, train_y, likelihood, gp_cfg: DictConfig, **kermut_params
    ):
        super(ExactGPKermut, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = hydra.utils.instantiate(
            gp_cfg.model, **kermut_params, **gp_cfg.kernel_params
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPModelKermutSequential(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, **kermut_params):
        super(ExactGPModelKermutSequential, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = KermutHellingerKernel(**kermut_params)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gp(
    model, likelihood, x: torch.tensor, y: torch.tensor, max_iter: int, **kwargs
):
    """Routine to train GP model using exact inference.

    Args:
        model: GP model
        likelihood: Likelihood
        x (torch.tensor): Input values
        y (torch.tensor): Target values
        max_iter (int): Number of iterations to train for
        **kwargs: Additional arguments to pass to kernel

    """

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(max_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(x, **kwargs)
        # Calc loss and backprop gradients
        loss = -mll(output, y)
        loss.backward()
        print(f"Iter {i + 1}/{max_iter} - Loss: {loss.item()}")
        optimizer.step()
    model.eval()
    return model
