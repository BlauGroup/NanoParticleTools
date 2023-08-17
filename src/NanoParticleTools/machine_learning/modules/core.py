import torch
from torch import nn
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
                 mid_dim: list[int] | int,
                 dropout_probability: float,
                 activation_module: torch.nn.Module = torch.nn.SiLU):
        super().__init__()
        if isinstance(mid_dim, int):
            mid_dim = [mid_dim]

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


class LazyNonLinearMLP(torch.nn.Module):

    def __init__(self,
                 out_dim: int,
                 mid_dim: list[int] | int,
                 dropout_probability: float,
                 activation_module: torch.nn.Module = torch.nn.SiLU):
        super().__init__()
        if isinstance(mid_dim, int):
            mid_dim = [mid_dim]

        _nn = []
        _nn.append(nn.LazyLinear(mid_dim[0]))
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
            self.running_mean: torch.Tensor | None
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
            self.num_batches_tracked: torch.Tensor | None
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
