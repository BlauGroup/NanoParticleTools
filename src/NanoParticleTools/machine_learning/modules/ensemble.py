from typing import Any
import torch
from torch_geometric.data import Data
from collections.abc import Callable


class EnsembleModel(torch.nn.Module):
    """
    Module that runs multiple models in parallel
    and gets statistics on the output
    """

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def __getattr__(self, name: str) -> Any:
        if name not in dir(self):
            return getattr(self.models[0], name)
        else:
            return super().__getattr__(name)

    def ensemble_forward(self, data: Data,
                         fn: Callable) -> dict[str, torch.Tensor]:
        output = []
        for module in self.models:
            y_hat = getattr(module, fn)(module, data)
            output.append(y_hat)

        x = torch.cat(output, dim=-1)
        return {'y': x, 'y_hat': x.mean(-1), 'std': x.std(-1)}

    def evaluate_step(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        output = []
        for module in self.models:
            y_hat = module.predict_step(data)
            output.append(y_hat)

        x = torch.cat(output, dim=-1)
        y_hat = x.mean(-1)
        loss = self.loss_function(y_hat, data.log_y)
        return y_hat, loss

    def predict_step(
            self,
            data: Data,
            return_stats=False) -> torch.Tensor | dict[str, torch.Tensor]:
        output = []
        for module in self.models:
            y_hat = module.predict_step(data)
            output.append(y_hat)

        x = torch.cat(output, dim=-1)
        if return_stats:
            return {'y': x, 'y_hat': x.mean(-1), 'std': x.std(-1)}
        else:
            return x.mean(-1)
