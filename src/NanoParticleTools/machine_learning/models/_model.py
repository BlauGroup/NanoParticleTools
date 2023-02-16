import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Callable, Optional, Union, List
from torch.optim.lr_scheduler import ReduceLROnPlateau

class SpectrumModelBase(pl.LightningModule):
    def __init__(self,
                 learning_rate: Optional[float]=1e-5,
                 l2_regularization_weight: float = 0,
                 lr_scheduler: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler]]=None,
                 loss_function: Optional[Callable[[List, List], float]] = F.mse_loss,
                 additional_metadata: Optional[dict] = {},
                 optimizer_type: Optional[str] = None,
                 augment_loss: Optional[bool] = False,
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
        self.augment_loss = augment_loss

        self.save_hyperparameters()

    def configure_optimizers(self) -> Union[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
        """
        """ 
        # Default to the adam optimizer               
        if self.optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_regularization_weight)
        elif self.optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_regularization_weight)
        else:
            # Default to amsgrad
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_regularization_weight, amsgrad=True)

        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
        else:
            lr_scheduler = None

        if isinstance(lr_scheduler, ReduceLROnPlateau):
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
              loss_prefix, 
              batch = None, 
              batch_idx = None):
        y_hat, loss = self._evaluate_step(batch)

        # Determine the batch size
        if hasattr(batch, 'batch'):
            batch_size = batch.batch[-1]
        else:
            batch_size = 1

        # Log the loss
        metric_dict = {f'{loss_prefix}_loss': loss}
        if loss_prefix != 'train':
            # For the validation and test sets, log additional metrics
            metric_dict[f'{loss_prefix}_mse'] = F.mse_loss(y_hat, batch.log_y)
            metric_dict[f'{loss_prefix}_mae'] = F.l1_loss(y_hat, batch.log_y)
            metric_dict[f'{loss_prefix}_huber'] = F.huber_loss(y_hat, batch.log_y)
            metric_dict[f'{loss_prefix}_hinge'] = F.hinge_embedding_loss(y_hat, batch.log_y)
            metric_dict[f'{loss_prefix}_cos_sim'] = F.cosine_similarity(y_hat, batch.log_y, 1).mean(0)
            # These seem to match up with the mse loss, so we don't waste time computing them
            # metric_dict[f'{loss_prefix}_p1'] = F.pairwise_distance(y_hat, batch.log_y, p=1).mean(0)
            # metric_dict[f'{loss_prefix}_p2'] = F.pairwise_distance(y_hat, batch.log_y, p=2).mean(0)
            # metric_dict[f'{loss_prefix}_p3'] = F.pairwise_distance(y_hat, batch.log_y, p=3).mean(0)

        self.log_dict(metric_dict, batch_size=batch_size)
        return loss, metric_dict

    def training_step(self, 
                      batch = None, 
                      batch_idx = None):
        loss, _ = self._step('train', batch, batch_idx)
        return loss
        
    def validation_step(self, 
                        batch = None, 
                        batch_idx = None):
        loss, _ = self._step('val', batch, batch_idx)
        return loss
        
    def test_step(self, 
                  batch = None, 
                  batch_idx = None):
        loss, _ = self._step('test', batch, batch_idx)
        return loss
        
    def predict_step(self, 
                     batch = None, 
                     batch_idx = None):
        pred, _ = self._evaluate_step(batch)
        return pred