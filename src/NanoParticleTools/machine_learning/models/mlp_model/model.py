from torch_geometric.data import Batch, Data, HeteroData
from NanoParticleTools.machine_learning.core.model import SpectrumModelBase
from NanoParticleTools.machine_learning.modules import NonLinearMLP

import pytorch_lightning as pl
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data


class MLPSpectrumModel(SpectrumModelBase):

    def __init__(self,
                 max_layers: int = 4,
                 n_dopants: int = 3,
                 n_output_nodes: int = 1,
                 nn_layers: list[int] | None = None,
                 dropout_probability: float = 0,
                 use_volume: bool = True,
                 **kwargs):
        if nn_layers is None:
            nn_layers = [128]

        super().__init__(**kwargs)

        self.use_volume = use_volume
        self.dropout_probability = dropout_probability

        self.n_dopants = n_dopants
        self.max_layers = max_layers
        self.n_output_nodes = n_output_nodes
        self.n_layers = len(nn_layers)
        self.nn_layers = nn_layers

        self.vol_factor = nn.Parameter(torch.tensor(1e6).float())
        self.conc_factor = nn.Parameter(torch.tensor(100).float())

        # Build the Feed Forward Neural Network Architecture
        self.nn = NonLinearMLP(self.n_input_nodes, n_output_nodes, nn_layers,
                               dropout_probability, nn.SiLU)

        self.save_hyperparameters()

    @property
    def n_input_nodes(self):
        if self.use_volume:
            return (self.n_dopants + 2) * self.max_layers
        else:
            return (self.n_dopants + 1) * self.max_layers

    def forward(self, x, radii, radii_without_zero, **kwargs):
        x = x * self.conc_factor

        if self.use_volume:
            volume = 4 / 3 * torch.pi * (radii[..., 1:]**3 -
                                         radii[..., :-1]**3)

            # Normalize the volume
            volume = volume / self.vol_factor

            # Concatenate the tensors to get the input
            input = torch.cat([x, radii_without_zero, volume], dim=1)
        else:
            # Concatenate the tensors to get the input
            input = torch.cat([x, radii_without_zero], dim=1)

        # Make the prediction
        out = self.nn(input)
        return out

    def evaluate_step(self, batch: Data | Batch,
                      **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates the model on a batch of data. This is used in training and validation.
        """
        x = batch.x
        radii = batch.radii
        radii_without_zero = batch.radii_without_zero

        y_hat = self.forward(x, radii, radii_without_zero, **kwargs)

        loss = self.loss_function(y_hat, batch.log_y)

        return y_hat, loss
