from gpytorch.kernels import Kernel, RBFKernel, MaternKernel
from typing import Optional, Literal


class SequenceKernel(Kernel):
    def __init__(
        self,
        kernel_type: Literal["RBF", "Matern"],
        nu: Optional[float] = None,
        **kwargs,
    ):
        super(SequenceKernel, self).__init__(**kwargs)
        self.kernel_type = kernel_type
        self.nu = nu
        self.base_kernel = self._create_kernel(kernel_type, nu, **kwargs)

    def _create_kernel(self, kernel_type, nu: Optional[float] = None, **params):
        if kernel_type == "RBF":
            return RBFKernel(**params)
        elif kernel_type == "Matern":
            assert nu in [0.5, 1.5, 2.5]
            return MaternKernel(nu=nu, **params)
        else:
            raise NotImplementedError

    def forward(self, x1, x2, diag=False, **params):
        return self.base_kernel.forward(x1, x2, diag=diag, **params)

    @property
    def is_stationary(self) -> bool:
        return True
