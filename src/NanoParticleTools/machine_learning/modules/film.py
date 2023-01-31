from typing import Optional
import torch
from torch import nn

def film_layer(x, gamma, beta):
    return torch.add(torch.mul(x, gamma), beta)

class FiLMLayer(nn.Module):
    def __init__(self,
                 conditioning_dim: int,
                 feature_dim: int,
                 intermediate_dim: Optional[int] = None):
        super().__init__()
        if intermediate_dim is not None:
            self.film_gen = NonLinearFiLMGenerator(conditioning_dim, feature_dim, intermediate_dim)
        else:
            self.film_gen = FiLMGenerator(conditioning_dim, feature_dim)
    
    def forward(self, x, z):
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
        
    def forward(self, z):
        gamma = self.gamma_mlp(z)
        beta = self.beta_mlp(z)
        
        return gamma, beta

class NonLinearFiLMGenerator(torch.nn.Module):
    def __init__(self, 
                 conditioning_dim,
                 feature_dim,
                 intermediate_dim: Optional[int] = None
                 ):
        super().__init__()
        self.gamma_mlp = nn.Sequential(nn.Linear(conditioning_dim, intermediate_dim), 
                                        nn.ReLU(),
                                        nn.Linear(intermediate_dim, feature_dim))
        self.beta_mlp = nn.Sequential(nn.Linear(conditioning_dim, intermediate_dim), 
                                        nn.ReLU(),
                                        nn.Linear(intermediate_dim, feature_dim))
        
    def forward(self, z):
        gamma = self.gamma_mlp(z)
        beta = self.beta_mlp(z)
        
        return gamma, beta
