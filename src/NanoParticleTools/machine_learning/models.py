from multiprocessing.sharedctypes import Value
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
                 n_hidden_layers: Union[int, float] = 2,
                 n_hidden_1: Union[int, float] = 128,
                 n_hidden_2: Union[int, float] = 128,
                 n_hidden_3: Union[int, float] = 128,
                 n_hidden_4: Union[int, float] = 128,
                 learning_rate: Optional[float]=1e-5,
                 lr_scheduler: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler]]=None,
                 loss_function: Optional[Callable[[List, List], float]] = F.mse_loss,
                 l2_regularization_weight: float = 0,
                 dropout_probability: float = 0, 
                 optimizer_type: str = 'SGD',
                 **kwargs):
        super().__init__()
        
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.l2_regularization_weight = l2_regularization_weight
        self.dropout_probability = dropout_probability

        self.n_input_nodes = n_input_nodes
        self.n_output_nodes = n_output_nodes
        # Make sure to cast the following floats into ints, allowing for BO to tune the # of hidden nodes
        self.n_hidden_layers = int(n_hidden_layers)
        self.n_hidden_1 = int(n_hidden_1) if self.n_hidden_layers >= 1 else 0
        self.n_hidden_2 = int(n_hidden_2) if self.n_hidden_layers >= 2 else 0
        self.n_hidden_3 = int(n_hidden_3) if self.n_hidden_layers >= 3 else 0
        self.n_hidden_4 = int(n_hidden_4) if self.n_hidden_layers >= 4 else 0

        self.save_hyperparameters()
        hidden_sizes = [self.n_hidden_1, self.n_hidden_2, self.n_hidden_3, self.n_hidden_4]

        current_n_nodes = self.n_input_nodes
        layers = []
        for i in range(self.n_hidden_layers):
            layers.extend(self._get_layer(current_n_nodes, hidden_sizes[i]))
            current_n_nodes = hidden_sizes[i]
        layers.append(nn.Linear(current_n_nodes, self.n_output_nodes))
        # Add a final softplus to constrain outputs to be always positive
        layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function

    def _get_layer(self, n_input_nodes, n_output_nodes):
        _layers = []
        _layers.append(nn.Linear(n_input_nodes, n_output_nodes))
        _layers.append(nn.Softplus())
        if self.dropout_probability > 0:
            _layers.append(nn.Dropout(self.dropout_probability))       
        return _layers

    
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

    def forward(self, x):
        return self.net(x)
    
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

    def predict_step_with_grad(self,
                               batch,
                               batch_idx):
        x, y = batch
        x = torch.tensor(x, requires_grad=True)
        pred = self(x)