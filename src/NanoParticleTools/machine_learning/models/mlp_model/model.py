import torch
from torch.utils import data
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Callable, Optional, Union, List
import numpy as np
from .._model import SpectrumModelBase

class SpectrumModel(SpectrumModelBase):
    def __init__(self, 
                 **kwargs):
        super().__init__(**kwargs)
    
    def describe(self):
        """
        Output a name that captures the hyperparameters used
        """
        descriptors = []

        # Type of optimizer
        if self.optimizer_type.lower() == 'sgd':
            descriptors.append('sgd')
        else:
            descriptors.append('adam')
        
        # Number of nodes/layers
        nodes = []
        nodes.append(self.n_input_nodes)
        nodes.extend(self.n_hidden_nodes)
        nodes.append(self.n_output_nodes)
        descriptors.append('-'.join([str(i) for i in nodes]))

        descriptors.append(f'lr-{self.learning_rate}')
        descriptors.append(f'dropout-{self.dropout_probability:.2f}')
        descriptors.append(f'l2_reg-{self.l2_regularization_weight:.2E}')
        
        return '_'.join(descriptors)

    def forward(self, data):
        x = data.x

        out = self.nn(x)
        return out
    
    def configure_optimizers(self) -> Union[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
        """
        """ 
        # Default to the adam optimizer               
        if self.optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_regularization_weight)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_regularization_weight)

        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
        return [optimizer], [lr_scheduler]

    def _evaluate_step(self, 
                       data):
        y_hat = self(data)
        loss = self.loss_function(y_hat, data.y)
        return y_hat, loss