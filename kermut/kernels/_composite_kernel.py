from typing import Literal, Tuple

import hydra
import torch
from gpytorch import Module
from gpytorch.kernels import ScaleKernel
from omegaconf import DictConfig
from torch import LongTensor, Tensor

from ._sequence_kernel import SequenceKernel
from ._structure_kernel import StructureKernel


class CompositeKernel(Module):
    """Composite kernel for Kermut GP.

    The combination can be done in three ways: weighted sum, addition, or multiplication.
    For weighted sum, a learnable parameter pi controls the contribution of each kernel.
    For addition, both kernels are independently scaled.
    For multiplication, the product kernel is scaled.

    Args:
        structure_kernel (DictConfig): Configuration for the structure kernel.
        sequence_kernel (DictConfig): Configuration for the sequence kernel.
        composition (Literal["weighted_sum", "add", "multiply"]): How to combine the kernels.
            Default is "weighted_sum".
        **kwargs: Additional keyword arguments passed to structure_kernel instantiation.

    Attributes:
        structure_kernel (StructureKernel): The instantiated structure kernel.
        sequence_kernel (SequenceKernel): The instantiated sequence kernel.
        composition (str): The method used to combine kernels.
        pi (Parameter, optional): Learnable weight parameter for weighted sum composition.
        scale_kernel (ScaleKernel, optional): Scaling kernel for multiply composition.
    """

    def __init__(
        self,
        structure_kernel: DictConfig,
        sequence_kernel: DictConfig,
        composition: Literal["weighted_sum", "add", "multiply"] = "weighted_sum",
        **kwargs,
    ):
        super().__init__()

        self.structure_kernel: StructureKernel = hydra.utils.instantiate(structure_kernel, **kwargs)
        self.sequence_kernel: SequenceKernel = hydra.utils.instantiate(sequence_kernel)

        self.composition = composition
        match composition:
            case "weighted_sum":
                # This formulation follows NeurIPS manuscript.
                self.register_parameter("pi", torch.nn.Parameter(torch.tensor(0.5)))
                self.structure_kernel = ScaleKernel(self.structure_kernel)
            case "add":
                self.structure_kernel = ScaleKernel(self.structure_kernel)
                self.sequence_kernel = ScaleKernel(self.sequence_kernel)
            case "multiply":
                self.scale_kernel = ScaleKernel()

    def forward(
        self,
        x1: Tuple[LongTensor, Tensor],
        x2: Tuple[LongTensor, Tensor] = None,
        **params,
    ) -> Tensor:
        if x2 is None:
            x2 = x1

        x1_toks, x1_emb = x1
        x2_toks, x2_emb = x2

        k_struct = self.structure_kernel(x1_toks, x2_toks, **params)
        k_seq = self.sequence_kernel(x1_emb, x2_emb, **params)

        match self.composition:
            case "weighted_sum":
                return k_struct * torch.sigmoid(self.pi) + k_seq * (1 - torch.sigmoid(self.pi))
            case "add":
                return k_struct + k_seq
            case "multiply":
                return self.scale_kernel(k_struct * k_seq)
