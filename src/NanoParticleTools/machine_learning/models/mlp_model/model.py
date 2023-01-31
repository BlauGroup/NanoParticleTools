import torch
from torch.utils import data
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Callable, Optional, Union, List
import numpy as np
from .._model import SpectrumModelBase

class MLPSpectrumModel(SpectrumModelBase):
    def __init__(self, 
                 n_input_nodes: int,
                 n_output_nodes: Optional[int] = 600,
                 dopants: Union[list, dict] = ['Yb', 'Er', 'Nd'],
                 nn_layers: Optional[List[int]] = [128],
                 dropout_probability: float = 0, 
                 **kwargs):
        super().__init__(**kwargs)

        self.dropout_probability = dropout_probability

        if isinstance(dopants, list):
            self.dopant_map = {key:i for i, key in enumerate(dopants)}
        elif isinstance(dopants, dict):
            self.dopant_map = dopants
        else:
            raise ValueError('Expected dopants to be of type list or dict')
        
        self.dopants = dopants

        self.n_input_nodes = n_input_nodes
        self.n_output_nodes = n_output_nodes
        self.n_layers = len(nn_layers)
        self.nn_layers = nn_layers

        # Build the Feed Forward Neural Network Architecture
        current_n_nodes = self.n_input_nodes
        layers = []
        for i, n_nodes in enumerate(self.nn_layers):
            layers.extend(self._get_layer(current_n_nodes, n_nodes))
            current_n_nodes = n_nodes
        layers.append(nn.Linear(current_n_nodes, self.n_output_nodes))
        # Add a final ReLU to constrain outputs to be always positive
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

    def forward(self, x, **kwargs):
        out = self.nn(x)
        return out