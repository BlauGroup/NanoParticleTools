from typing import Union, Any
from torch import Tensor, nn
import torch
from torch.nn.modules.module import Module


class EnsembleModel(nn.Module):
    """
    Module that runs multiple models in parallel
    and gets statistics on the output
    """
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def __getattr__(self, name: str) -> Any:
        if name not in dir(self):
            return getattr(self.models[0], name)
        else:
            return super().__getattr__(name)

    def ensemble_forward(self, data, fn):
        output = []
        for module in self.models:
            y_hat = getattr(module, fn)(module, data)
            output.append(y_hat)

        x = torch.cat(output, dim=-1)
        return {'y': x, 'y_hat': x.mean(-1), 'std': x.std()}

    def _evaluate_step(self, data):
        output = []
        for module in self.models:
            y_hat = module.predict_step(data)
            output.append(y_hat)

        x = torch.cat(output, dim=-1)
        y_hat = x.mean(-1)
        loss = self.loss_function(y_hat, data.log_y)
        return y_hat, loss

    def predict_step(self, data, return_stats=False):
        output = []
        for module in self.models:
            y_hat = module.predict_step(data)
            output.append(y_hat)

        x = torch.cat(output, dim=-1)
        if return_stats:
            return {'y': x, 'y_hat': x.mean(-1), 'std': x.std()}
        else:
            return x.mean(-1)
