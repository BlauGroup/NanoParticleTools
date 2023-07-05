import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Any, Callable, Optional, Union, List
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, Batch


class SpectrumModelBase(pl.LightningModule):
    """
    This is the base class for all spectrum models. It cannot be used directly,
    since it does not implement the `forward` method.
    """
    def __init__(self,
                 learning_rate: Optional[float] = 1e-5,
                 l2_regularization_weight: float = 0,
                 lr_scheduler: Optional[Callable[[torch.optim.Optimizer],
                                                 torch.optim.lr_scheduler._LRScheduler]] = None,
                 loss_function: Optional[Callable[[List, List], float]] = F.mse_loss,
                 additional_metadata: Optional[dict] = {},
                 optimizer_type: Optional[str] = None,
                 **kwargs):

        super().__init__()

        if optimizer_type is None:
            self.optimizer_type = 'amsgrad'
        else:
            self.optimizer_type = optimizer_type

        self.learning_rate = learning_rate
        self.l2_regularization_weight = l2_regularization_weight
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
        self.additional_metadata = additional_metadata

        self.save_hyperparameters()

    def configure_optimizers(self) -> Union[List[torch.optim.Optimizer],
                                            List[torch.optim.lr_scheduler._LRScheduler]]:
        """
        Configures optimizers and learning rate schedulers for the model.

        Uses the `optimizer_type` attribute of self to determine which optimizer to use.
        Uses the `lr_scheduler` attribute of self to determine which learning rate scheduler to use.

        Set these in the constructor of the model.
        """
        if self.optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.learning_rate,
                                        weight_decay=self.l2_regularization_weight)
        elif self.optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.learning_rate,
                                         weight_decay=self.l2_regularization_weight)
        else:
            # Default to amsgrad
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.learning_rate,
                                         weight_decay=self.l2_regularization_weight,
                                         amsgrad=True)

        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
        else:
            lr_scheduler = None

        if isinstance(lr_scheduler, ReduceLROnPlateau):
            # For a ReduceLROnPlateau scheduler, we treat it specially, since it needs to monitor
            # a metric to determine when to reduce the learning rate.
            return [optimizer], [{"scheduler": lr_scheduler,
                                  "monitor": "val_loss",
                                  "frequency": 1,
                                  "strict": True}]
        else:
            return [optimizer], [lr_scheduler]

    def _evaluate_step(self,
                       data):
        y_hat = self(**data.to_dict())

        loss = self.loss_function(y_hat, data.log_y)
        return y_hat, loss

    def _step(self,
              prefix: str,
              batch: Data | Batch,
              batch_idx: int | None = None,
              log: bool = True):
        y_hat, loss = self._evaluate_step(batch)

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
            metric_dict[f'{prefix}_huber'] = F.huber_loss(y_hat, batch.log_y)
            metric_dict[f'{prefix}_hinge'] = F.hinge_embedding_loss(y_hat, batch.log_y)
            metric_dict[f'{prefix}_cos_sim'] = F.cosine_similarity(y_hat, batch.log_y, 1).mean(0)

        if log:
            self.log_dict(metric_dict, batch_size=batch_size)
        return loss, metric_dict

    def training_step(self,
                      batch: Data | Batch,
                      batch_idx: int | None = None) -> torch.Tensor:
        loss, _ = self._step('train', batch, batch_idx)
        return loss

    def validation_step(self,
                        batch: Data | Batch,
                        batch_idx: int | None = None) -> torch.Tensor:
        loss, _ = self._step('val', batch, batch_idx)
        return loss

    def test_step(self,
                  batch: Data | Batch,
                  batch_idx: int | None = None) -> torch.Tensor:
        loss, _ = self._step('test', batch, batch_idx)
        return loss

    def predict_step(self,
                     batch: Data | Batch,
                     batch_idx: int | None = None) -> torch.Tensor:
        """
        Make a prediction for a batch of data.

        Args:
            batch (_type_): _description_
            batch_idx (int | None, optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: _description_
        """
        y_hat = self(**batch.to_dict())
        return y_hat
