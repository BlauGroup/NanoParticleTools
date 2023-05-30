import torch
from torch import nn
from typing import Optional, List, Union
import numpy as np


class ParallelModule(nn.Sequential):
    """
    A module that runs multiple modules in parallel and concatenates the output
    """

    def __init__(self, *args):
        super(ParallelModule, self).__init__(*args)

    def forward(self, input):
        output = []
        for module in self:
            output.append(module(input))
        return torch.cat(output, dim=1)
    

class NonLinearMLP(torch.nn.Module):

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 mid_dim: Union[List[int], int],
                 dropout_probability: float,
                 activation_module: Optional[torch.nn.Module] = None):
        super().__init__()
        if isinstance(mid_dim, int):
            mid_dim = [mid_dim]

        if activation_module is None:
            activation_module = torch.nn.SiLU

        _nn = []
        _nn.append(nn.Linear(in_dim, mid_dim[0]))
        if activation_module is not None:
            _nn.append(activation_module())
        if dropout_probability > 0:
            _nn.append(nn.Dropout(dropout_probability))
        current_dim = mid_dim[0]
        for _dim in mid_dim[1:]:
            _nn.append(nn.Linear(current_dim, _dim))
            if activation_module is not None:
                _nn.append(activation_module())
            if dropout_probability > 0:
                _nn.append(nn.Dropout(dropout_probability))
            current_dim = _dim
        _nn.append(nn.Linear(current_dim, out_dim))
        self.mlp = nn.Sequential(*_nn)

    def forward(self, x):
        return self.mlp(x)


class BatchScaling(nn.Module):

    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 track_running_stats: bool = True,
                 device=None,
                 dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats

        if self.track_running_stats:
            self.register_buffer('running_mean',
                                 torch.zeros(num_features, **factory_kwargs))
            # self.register_buffer('running_var', torch.ones(num_features, **factory_kwargs))
            self.running_mean: Optional[torch.Tensor]
            # self.running_var: Optional[torch.Tensor]
            self.register_buffer(
                'num_batches_tracked',
                torch.tensor(0,
                             dtype=torch.long,
                             **{
                                 k: v
                                 for k, v in factory_kwargs.items()
                                 if k != 'dtype'
                             }))
            self.num_batches_tracked: Optional[torch.Tensor]
        else:
            self.register_buffer("running_mean", None)
            # self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            # running_mean/running_var/num_batches... are registered at runtime depending
            # if self.track_running_stats is on
            # Initialize the mean to ones, such that scaling is performed if not trained
            self.running_mean.fill_(1)  # type: ignore[union-attr]
            # self.running_var.fill_(1)  # type: ignore[union-attr]
            self.num_batches_tracked.zero_(
            )  # type: ignore[union-attr,operator]

    def reset_parameters(self) -> None:
        self.reset_running_stats()

    def forward(self, input: torch.Tensor):
        if input.size(0) > 1:
            if self.training and self.track_running_stats:
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                with torch.no_grad():
                    self.running_mean = (
                        1 - self.momentum
                    ) * self.running_mean + self.momentum * input.mean(0)

        return input / (self.running_mean + self.eps)


class BesselBasis(torch.nn.Module):
    """
    Adapted from MACE Implementation

    ###########################################################################################
    # Radial basis and cutoff
    # Authors: Ilyes Batatia, Gregor Simm
    # This program is distributed under the MIT License (see MIT.md)
    ###########################################################################################

    Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing for Molecular Graphs;
    ICLR 2020. Equation (7)
    """

    def __init__(self, r_max: float, num_basis=8, trainable=False):
        super().__init__()

        bessel_weights = (np.pi / r_max * torch.linspace(
            start=1.0,
            end=num_basis,
            steps=num_basis,
            dtype=torch.get_default_dtype(),
        ))
        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "prefactor",
            torch.tensor(np.sqrt(2.0 / r_max),
                         dtype=torch.get_default_dtype()),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:  # [..., 1]
        x = x + 0.01  # Shifted to ensure it is not inf at 0
        numerator = torch.sin(self.bessel_weights * x)  # [..., num_basis]
        return self.prefactor * (numerator / (x))

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={len(self.bessel_weights)}, "
            f"trainable={self.bessel_weights.requires_grad})")


class GeometricEmbedding(nn.Module):
    r"""
    Here, we treat the system as

    It is possible to extend the Bessel Basis from 1D to 2D or 3D. In the DimeNet paper,
    they obtain the 1D Bessel Basis from the simplfication of l=0, m=0. The 2D basis they
    only propose restricts m=0. Therefore we can similarly get a 3D basis function without
    any restrictions on l and m (in $\mathbf{\gamma}^l_m$)
    """

    def __init__(self, r_max, embed_dim):
        super().__init__()
        self.bessel_fn = BesselBasis(r_max, embed_dim)

    def forward(self, r):
        p = self.bessel_fn(r.reshape(-1, 1)).reshape(*r.shape, -1)
        return p[:, 1] - p[:, 0]
