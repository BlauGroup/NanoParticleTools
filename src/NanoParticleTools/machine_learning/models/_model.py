import torch
from torch.utils import data
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Callable, Optional, Union, List
import numpy as np


class SpectrumModelBase(pl.LightningModule):
    def __init__(self,
                 n_input_nodes: int,
                 n_output_nodes: Optional[int]=20,
                 nn_layers: Optional[List[int]] = [128],
                 dopants: Union[list, dict] = ['Yb', 'Er', 'Nd'],
                 learning_rate: Optional[float]=1e-5,
                 lr_scheduler: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler]]=None,
                 loss_function: Optional[Callable[[List, List], float]] = F.mse_loss,
                 l2_regularization_weight: float = 0,
                 dropout_probability: float = 0, 
                 optimizer_type: str = 'SGD',
                 additional_metadata: Optional[dict] = {},
                 **kwargs):
        super().__init__()

        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.l2_regularization_weight = l2_regularization_weight
        self.dropout_probability = dropout_probability
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
        self.additional_metadata = additional_metadata


        self.n_layers = len(nn_layers)
        self.nn_layers = nn_layers
        self.n_input_nodes = n_input_nodes
        self.n_output_nodes = n_output_nodes
        
        if isinstance(dopants, list):
            self.dopant_map = {key:i for i, key in enumerate(dopants)}
        elif isinstance(dopants, dict):
            self.dopant_map = dopants
        else:
            raise ValueError('Expected dopants to be of type list or dict')
        
        self.dopants = dopants
        
        # Build the Feed Forward Neural Network Architecture
        current_n_nodes = self.n_input_nodes
        layers = []
        for i, n_nodes in enumerate(self.nn_layers):
            layers.extend(self._get_layer(current_n_nodes, n_nodes))
            current_n_nodes = n_nodes
        layers.append(nn.Linear(current_n_nodes, self.n_output_nodes))
        # Add a final softplus to constrain outputs to be always positive
        layers.append(nn.ReLU())
        self.nn = nn.Sequential(*layers)
        
        self.save_hyperparameters()

    def _get_layer(self, n_input_nodes, n_output_nodes):
        _layers = []
        _layers.append(nn.Linear(n_input_nodes, n_output_nodes))
        _layers.append(nn.ReLU())
        if self.dropout_probability > 0:
            _layers.append(nn.Dropout(self.dropout_probability))       
        return _layers

    def configure_optimizers(self) -> Union[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
        """
        """ 
        # Default to the adam optimizer               
        if self.optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_regularization_weight)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_regularization_weight, amsgrad=True)

        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
        return [optimizer], [lr_scheduler]

    def forward(self, types, volumes, compositions):
        pass

    def _evaluate_step(self, 
                       batch):
        pass

    def training_step(self, 
                      batch, 
                      batch_idx):
        _, loss = self._evaluate_step(batch)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, 
                        batch, 
                        batch_idx):
        _, loss = self._evaluate_step(batch)
        self.log('val_loss', loss)
        return loss
        
    def test_step(self, 
                  batch, 
                  batch_idx):
        _, loss = self._evaluate_step(batch)
        self.log('test_loss', loss)
        return loss
        
    def predict_step(self, 
                     batch, 
                     batch_idx):
        pred, _ = self._evaluate_step(batch)
        return pred

    def predict_step_with_grad(self,
                               batch,
                               batch_idx):
        x, y = batch
        x = torch.tensor(x, requires_grad=True)
        pred = self(x)
        return pred