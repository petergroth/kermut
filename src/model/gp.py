import gpytorch
import torch
import pandas as pd


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gp(model, likelihood, x: torch.tensor, y: torch.tensor, max_iter: int):
    """Routine to train GP model using exact inference.

    Args:
        model: GP model
        likelihood: Likelihood
        x (torch.tensor): Input values
        y (torch.tensor): Target values
        max_iter (int): Number of iterations to train for

    """

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(max_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(x)
        # Calc loss and backprop gradients
        loss = -mll(output, y)
        loss.backward()
        print(f"Iter {i + 1}/{iter} - Loss: {loss.item()}")
        optimizer.step()
    model.eval()
    return model
