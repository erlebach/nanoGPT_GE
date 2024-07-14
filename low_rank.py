# import beartype
# from torch.nn import functional as F
from jaxtyping import Float
import torch
from torch import nn
from torch import Tensor as T


class LowRankLinear(nn.Linear):
    """
    Extends a PyTorch linear layer with Low-Rank Adaptation (LoRA).
    LoRA adds two matrices to the layer, allowing for efficient training of large models.
    """

    def __init__(
        self, in_features: int, out_features: int, *args, r: int = 8, **kwargs
    ) -> None:
        super().__init__(in_features, out_features, *args, **kwargs)

        # could lead to slower convergence if not initialized properly
        # self.low_rank_matrix_B = nn.Parameter(torch.zeros(out_features, r))
        self.low_rank_matrix_B = nn.Parameter(torch.randn(out_features, r))
        self.low_rank_matrix_A = nn.Parameter(torch.randn(r, in_features))
        del self.weight

    def forward(
        self, input: Float[T, ["... in_features"]]
    ) -> Float[T, ["... out_features"]]:
        """
        output:
        """
        return (input @ self.low_rank_matrix_B) @ self.low_rank_matrix_A + self.bias
