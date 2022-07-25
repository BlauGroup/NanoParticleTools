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
                 n_hidden_layers: int = 2,
                 n_hidden_nodes: Optional[Union[int, List[int]]] = 128, 
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

        self.save_hyperparameters()

        if isinstance(n_hidden_nodes, int):
            n_hidden_nodes = [n_hidden_nodes for _ in range(n_hidden_layers)]
        elif isinstance(n_hidden_nodes, list) and len(n_hidden_nodes) != n_hidden_layers:
            raise ValueError(f'Specified n_hidden_layers = {n_hidden_layers}, but length of n_hidden is {len(n_hidden_nodes)}')
        else:
            raise ValueError('n_hidden is not of type int')
        
        self.n_input_nodes = n_input_nodes
        self.n_hidden_nodes = n_hidden_nodes
        self.n_output_nodes = n_output_nodes
        self.n_hidden_layers = n_hidden_layers
                    
        current_n_nodes = n_input_nodes
        layers = []
        for i, layer in enumerate(range(n_hidden_layers)):
            layers.append(nn.Linear(current_n_nodes, n_hidden_nodes[i]))
            layers.append(nn.Softplus())
            if dropout_probability > 0:
                layers.append(nn.Dropout(dropout_probability))
            current_n_nodes = n_hidden_nodes[i]
        layers.append(nn.Linear(current_n_nodes, n_output_nodes))
        # Add a final softplus to constrain outputs to be always positive
        layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
    
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