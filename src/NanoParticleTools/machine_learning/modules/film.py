from typing import Optional
import torch
from torch import nn
from typing import List, Tuple

@torch.jit.script
def film_layer(x, gamma, beta):
    return torch.add(torch.mul(x, gamma), beta)


class FiLMLayer(nn.Module):
    def __init__(self,
                 conditioning_dim: int,
                 feature_dim: int,
                 intermediate_dim: int | List[int] = None):
        super().__init__()
        if intermediate_dim is not None:
            self.film_gen = NonLinearFiLMGenerator(conditioning_dim, feature_dim, intermediate_dim)
        else:
            self.film_gen = FiLMGenerator(conditioning_dim, feature_dim)

    def forward(self,
                x: torch.Tensor,
                z: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film_gen(z)
        return film_layer(x, gamma, beta)


class FiLMGenerator(torch.nn.Module):
    def __init__(self,
                 conditioning_dim: int,
                 feature_dim: int
                 ):
        super().__init__()
        self.gamma_mlp = nn.Sequential(nn.Linear(conditioning_dim, feature_dim))
        self.beta_mlp = nn.Sequential(nn.Linear(conditioning_dim, feature_dim))

    def forward(self,
                z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma = self.gamma_mlp(z)
        beta = self.beta_mlp(z)

        return gamma, beta


class NonLinearFiLMGenerator(torch.nn.Module):
    def __init__(self,
                 conditioning_dim: int,
                 feature_dim: int,
                 intermediate_dim: int | List[int] = None
                 ):
        super().__init__()
        if isinstance(intermediate_dim, int):
            intermediate_dim = [intermediate_dim]
        
        self.gamma_mlp = self.get_mlp(conditioning_dim, intermediate_dim, feature_dim)
        self.beta_mlp = self.get_mlp(conditioning_dim, intermediate_dim, feature_dim)

    @staticmethod
    def get_mlp(in_dim: int,
                dims: List[int],
                out_dim: int) -> nn.Module:
        modules = []
        modules.append(nn.Linear(in_dim, dims[0]))
        modules.append(nn.ReLU())
        _in_dim = dims[0]
        for dim in dims[1:]:
            modules.append(nn.Linear(_in_dim, dim))
            modules.append(nn.ReLU())
            _in_dim = dim
        modules.append(nn.Linear(_in_dim, out_dim))

        return nn.Sequential(*modules)

    def forward(self,
                z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma = self.gamma_mlp(z)
        beta = self.beta_mlp(z)

        return gamma, beta
