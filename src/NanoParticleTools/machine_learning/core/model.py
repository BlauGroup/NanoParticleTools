from NanoParticleTools.machine_learning.util.learning_rate import ReduceLROnPlateauWithWarmup
from torch_geometric.data import Data, Batch
import pytorch_lightning as pl

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch import nn

from typing import Any, Callable, Optional, Union, List


class SpectrumModelBase(pl.LightningModule):
    """
    This is the base class for all spectrum models. It cannot be used directly,
    since it does not implement the `forward` method.
    """

    def __init__(self,
                 l2_regularization_weight: float = 0,
                 optimizer_type: Optional[str] = None,
                 learning_rate: Optional[float] = 1e-5,
                 lr_scheduler: torch.optim.lr_scheduler.
                 _LRScheduler = ReduceLROnPlateauWithWarmup,
                 lr_scheduler_kwargs: Optional[dict] = None,
                 loss_function: Optional[Callable[[List, List],
                                                  float]] = F.mse_loss,
                 additional_metadata: Optional[dict] = {},
                 **kwargs):
        """
        Args:
            learning_rate: The default learning rate for model training. The actual learning rate
                used may be different depending on the actions of the learning rate scheduler
            optimizer_type: The type of optimizer to use. options are 'sgd', 'adam', and 'amsgrad'.
                if 'amsgrad' is selected, the pytorch adam optimizer is used with the `amsgrad=True`
            l2_regularization_weight: The weight of the L2 regularization term in the loss function.
                This is passed to the torch optimizer
            lr_scheduler: The learning rate scheduler class to use.
            lr_scheduler_kwargs: The kwargs passed to the learning rate scheduler on initialization.
            loss_function: The loss function to use for backpropagation in training.
                MAE, MSE, and Cosine Similarity will be logged anyways.
            additional_metadata: Additional metadata which will be logged with the model to
                wandb.
        """
        super().__init__()

        if optimizer_type is None:
            optimizer_type = 'amsgrad'
        self.optimizer_type = optimizer_type

        if lr_scheduler_kwargs is None:
            lr_scheduler_kwargs = {}
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        self.learning_rate = learning_rate
        self.l2_regularization_weight = l2_regularization_weight
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
        self.additional_metadata = additional_metadata

        # Save hyperparameters (which logs to wandb)
        self.save_hyperparameters()

    def configure_optimizers(
        self
    ) -> Union[List[torch.optim.Optimizer],
               List[torch.optim.lr_scheduler._LRScheduler]]:
        """
        Configures optimizers and learning rate schedulers for the model.

        Uses the `optimizer_type` attribute of self to determine which optimizer to use.
        Uses the `lr_scheduler` attribute of self to determine which learning rate scheduler to use.

        Set these in the constructor of the model.
        """
        if self.optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.l2_regularization_weight)
        elif self.optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.l2_regularization_weight)
        else:
            # Default to amsgrad
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.l2_regularization_weight,
                amsgrad=True)

        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer=optimizer,
                                             **self.lr_scheduler_kwargs)
        else:
            lr_scheduler = None

        if isinstance(lr_scheduler, ReduceLROnPlateau):
            # For a ReduceLROnPlateau scheduler, we treat it specially, since it needs to monitor
            # a metric to determine when to reduce the learning rate.
            return [optimizer], [{
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
                "frequency": 1,
                "strict": True
            }]
        else:
            return [optimizer], [lr_scheduler]

    def predict_step(self, batch: Data | Batch, **kwargs) -> torch.Tensor:
        """
        Make a prediction for a batch of data.

        Args:
            batch: A single data point or a collated batch of data points.

        Returns:
            y_hat: The predicted value(s) for the batch
        """
        y_hat = self(**batch.to_dict())
        return y_hat

    def evaluate_step(self, batch: Data | Batch,
                      **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        A single forward pass of the data and loss calculation.

        Args:
            batch: A single data point or a collated batch of data points.

        Returns:
            y_hat: The predicted value(s) for the batch
            loss: The loss for the batch
        """
        y_hat = self(**batch.to_dict())

        loss = self.loss_function(y_hat, batch.log_y)
        return y_hat, loss

    def _step(self,
              prefix: str,
              batch: Data | Batch,
              log_to_wandb: bool = True,
              **kwargs) -> tuple[torch.Tensor, dict]:
        """
        Evaluate a single batch of data and compute all metrics for logging

        Args:
            prefix: The prefix to use for logging. Typically 'train', 'val', or 'test'
            batch: A single data point or a collated batch of data points.
            log_to_wandb: Whether to log the metrics to wandb

        Returns:
            loss: The loss for the batch
            metric_dict: A dictionary of metrics (mse, mae, cos_sim) for the batch
        """
        y_hat, loss = self.evaluate_step(batch)

        # Determine the batch size
        if batch.batch is not None:
            batch_size = batch.batch[-1]
        else:
            batch_size = 1

        # Log the loss
        metric_dict = {f'{prefix}_loss': loss}
        if prefix != 'train':
            # For the validation and test sets, log additional metrics
            metric_dict[f'{prefix}_mse'] = F.mse_loss(y_hat, batch.log_y)
            metric_dict[f'{prefix}_mae'] = F.l1_loss(y_hat, batch.log_y)
            metric_dict[f'{prefix}_cos_sim'] = F.cosine_similarity(
                y_hat, batch.log_y, 1).mean(0)

        if log_to_wandb:
            self.log_dict(metric_dict, batch_size=batch_size)
        return loss, metric_dict

    def training_step(self,
                      batch: Data | Batch,
                      batch_idx: int | None = None) -> torch.Tensor:
        """
        A single training step.

        Args:
            batch: A single data point or a collated batch of data points.
            batch_idx: The index of the batch

        Returns:
            loss: The loss for the batch
        """
        loss, _ = self._step('train', batch, batch_idx)
        return loss

    def validation_step(self,
                        batch: Data | Batch,
                        batch_idx: int | None = None) -> torch.Tensor:
        """
        A single validation step.

        Args:
            batch: A single data point or a collated batch of data points.
            batch_idx: The index of the batch

        Returns:
            loss: The loss for the batch
        """
        loss, _ = self._step('val', batch, batch_idx)
        return loss

    def test_step(self,
                  batch: Data | Batch,
                  batch_idx: int | None = None) -> torch.Tensor:
        """
        A single testing step.

        Args:
            batch: A single data point or a collated batch of data points.
            batch_idx: The index of the batch

        Returns:
            loss: The loss for the batch
        """
        loss, _ = self._step('test', batch, batch_idx)
        return loss
