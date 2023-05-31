from typing import Union, List
import torch


class QuantileLoss(torch.nn.Module):
    def __init__(self, 
                 p: Union[torch.Tensor, float]) -> None:
        super().__init__()

        self.p = p
        if isinstance(p, float):
            self.loss_fn = quantile_loss
        else:
            self.loss_fn = multi_quantile_loss

    def forward(self, prediction, target):
        return self.loss_fn(prediction, target, self.p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


@torch.jit.script
def quantile_loss(prediction: torch.FloatTensor,
                  target: torch.FloatTensor,
                  p: float):
    """
    The quantile loss
    
    p = 0.5 will set the MAE as the target
    p < 0.5 will incentivize underpredictions
    p > 0.5 will incentivize overpredictions
    """
    return torch.max(p * (target - prediction), (1 - p) * (prediction - target))


@torch.jit.script
def multi_quantile_loss(prediction: torch.FloatTensor,
                        target: torch.FloatTensor,
                        p: torch.FloatTensor) -> torch.BoolTensor:
    """
    The quantile loss using multiple quantiles. This is used for multi-output regression.
    
    p = 0.5 will set the MAE as the target
    p < 0.5 will incentivize underpredictions
    p > 0.5 will incentivize overpredictions
    """
    return torch.max(p * (target - prediction).unsqueeze(-1),
                     (torch.ones_like(p) - p) * (prediction - target).unsqueeze(-1))
