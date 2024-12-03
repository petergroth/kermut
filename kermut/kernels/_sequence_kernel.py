from typing import Literal, Optional

from gpytorch.kernels import Kernel, MaternKernel, RBFKernel


class SequenceKernel(Kernel):
    """Wrapper class for sequence (i.e., embedding) kernels that implements either RBF or
    Matérn kernels.

    Args:
        kernel_type: Type of kernel to use. Must be either "RBF" or "Matern".
        nu: The smoothness parameter for the Matérn kernel. Required if kernel_type
            is "Matern". Must be one of [0.5, 1.5, 2.5].
        **kwargs: Additional keyword arguments to be passed to the underlying kernel
            implementation.

    Raises:
        NotImplementedError: If kernel_type is not "RBF" or "Matern".
        AssertionError: If kernel_type is "Matern" and nu is not in [0.5, 1.5, 2.5].

    Attributes:
        kernel_type: The type of kernel being used ("RBF" or "Matern").
        nu: The smoothness parameter for Matérn kernel (None for RBF).
        base_kernel: The underlying kernel implementation (RBFKernel or MaternKernel).
    """

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
