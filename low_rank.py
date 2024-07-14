# import beartype
# from torch.nn import functional as F
from jaxtyping import Float
from pprint import pprint
import torch
from torch import nn
from torch import Tensor as T


class LowRankLinear(nn.Linear):
    """
    Replace the dense weight in a linear layer by the product of two rank-r matrices. 
    The hope is to reduce flops and memory when training large models. 
    """

    def __init__(
        self, in_features: int, out_features: int, *args, r: int = 40, **kwargs
    ) -> None:
        super().__init__(in_features, out_features, *args, **kwargs)
        print("kwargs"); pprint(kwargs)
        print(f"{in_features=}, {out_features=}")

        # could lead to slower convergence if not initialized properly
        # self.low_rank_matrix_B = nn.Parameter(torch.zeros(out_features, r))
        self.low_rank_matrix_B = nn.Parameter(torch.randn(in_features, r))
        self.low_rank_matrix_A = nn.Parameter(torch.randn(r, out_features))
        del self.weight

    def forward (self, input):
    #def forward(
        #self, input: Float[T, ["... in_features"]]
    #) -> Float[T, ["... out_features"]]:
        """
        output: x @ (B @ A) + bias
        """
        # print(f"{input.shape=}")
        # print(f"{self.low_rank_matrix_B.shape=}")
        # print(f"{self.low_rank_matrix_A.shape=}")
        # print(f"{self.bias=}")
        # quit()
        if self.bias is not None:
            return (input @ self.low_rank_matrix_B) @ self.low_rank_matrix_A + self.bias
        else:
            return (input @ self.low_rank_matrix_B) @ self.low_rank_matrix_A
