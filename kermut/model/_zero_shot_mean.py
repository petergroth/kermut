from gpytorch.means import ConstantMean
import torch


class ZeroShotMean(ConstantMean):
    def __init__(self):
        super().__init__()
        self.register_parameter(
            "zero_shot_scale", torch.nn.Parameter(torch.tensor(1.0))
        )

    def forward(self, x):
        constant = super().forward(x)
        return constant + self.zero_shot_scale * x
