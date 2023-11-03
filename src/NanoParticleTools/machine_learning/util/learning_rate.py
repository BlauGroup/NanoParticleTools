from torch.optim.lr_scheduler import (LinearLR, ExponentialLR, SequentialLR,
                                      ReduceLROnPlateau,
                                      EPOCH_DEPRECATION_WARNING)
import warnings
import torch

from typing import List


class WarmupSequentialLR(SequentialLR):

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        start_factor: float = 0.01,
        end_factor: float = 1,
        total_iters: int = 10,
        gamma: float = 0.995,
        milestones: List[int] = None,
    ):

        if milestones is None:
            milestones = [1000]

        linear_lr = LinearLR(optimizer,
                             start_factor=start_factor,
                             end_factor=end_factor,
                             total_iters=total_iters)

        exponential_lr = ExponentialLR(optimizer, gamma=gamma)
        super().__init__(optimizer, [linear_lr, exponential_lr],
                         milestones=milestones)


class ReduceLROnPlateauWithWarmup(ReduceLROnPlateau):
    """
    Modified ReduceLROnPlateau, which allows a warmup phase at
    the beginning of training.

    Having a warmup phase is useful so we can train the model
    with a high learning rate, without having to worry about
    the learning rate being too high and causing the model to
    diverge in the first few steps.
    """

    def __init__(self, warmup_epochs: int = 10, **kwargs):
        """
        Args:
            warmup_epochs: Number of epochs to use for the warmup phase.
            patience: Number of epochs to wait before reducing the learning rate.
            factor: Factor by which to reduce the learning rate.
        """
        # Set default patience
        if 'patience' not in kwargs:
            kwargs['patience'] = 100

        # Set default factor
        if 'factor' not in kwargs:
            kwargs['factor'] = 0.8

        super().__init__(**kwargs)
        self.warmup_epochs = warmup_epochs

        # Set the initial learning rate
        self._initial_step()

    def _initial_step(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = param_group['lr'] / 10
        self.initial_lr = [
            float(param_group['lr'])
            for i, param_group in enumerate(self.optimizer.param_groups)
        ]
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def step(self, metrics: float | torch.Tensor, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.last_epoch < self.warmup_epochs:
            self._increase_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _increase_lr(self, epoch):
        for i, (param_group, initial_lr) in enumerate(
                zip(self.optimizer.param_groups, self.initial_lr)):
            old_lr = float(param_group['lr'])
            new_lr = old_lr + initial_lr
            param_group['lr'] = new_lr
            if self.verbose:
                epoch_str = ("%.2f"
                             if isinstance(epoch, float) else "%.5d") % epoch
                print('Epoch {}: increasing learning rate'
                      ' of group {} to {:.4e}.'.format(epoch_str, i, new_lr))
