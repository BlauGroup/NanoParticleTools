import torch
from torch.utils import data
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Callable, Optional, Union, List

class SpectrumModel(pl.LightningModule):
    def __init__(self, 
                 n_input_nodes: Optional[int]=16, 
                 n_output_nodes: Optional[int]=20, 
                 n_hidden: Optional[int]=128, 
                 learning_rate: Optional[float]=1e-5,
                 lr_scheduler: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler]]=None,
                 loss_function: Optional[Callable[[List, List], float]] = F.mse_loss,
                 l2_regularization_weight: float = 0):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.l2_regularization_weight = l2_regularization_weight

        self.save_hyperparameters()

        self.net = nn.Sequential(nn.Linear(n_input_nodes, n_hidden),
                                 nn.Softplus(),
                                 nn.Linear(n_hidden, n_hidden),
                                 nn.Softplus(),
                                 nn.Linear(n_hidden, n_output_nodes))
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
    
    def forward(self, x):
        return self.net(x)
    
    def configure_optimizers(self) -> Union[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_regularization_weight)
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
        return [optimizer], [lr_scheduler]

    def training_step(self, 
                      batch, 
                      batch_idx):
        x, y = batch
        y_hat = self(x)
        train_loss = self.loss_function(y_hat, y)
        self.log('train_loss', train_loss)
        return train_loss
    
    def validation_step(self, 
                        batch, 
                        batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_function(y_hat, y)
        self.log('val_loss', val_loss)
        
    def test_step(self, 
                  batch, 
                  batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = self.loss_function(y_hat, y)
        self.log('test_loss', test_loss)
        
    def predict_step(self, 
                     batch, 
                     batch_idx):
        x, y = batch
        pred = self(x)
        return pred