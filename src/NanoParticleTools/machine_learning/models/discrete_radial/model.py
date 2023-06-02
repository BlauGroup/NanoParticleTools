import torch
from torch import nn
from torch.nn import functional as F
from typing import Callable, Union, Optional, List
import pytorch_lightning as pl
from torch_geometric import nn as pyg_nn
from torch_scatter.scatter import scatter
from NanoParticleTools.machine_learning.core import SpectrumModelBase


class CNNModel(SpectrumModelBase):

    def __init__(self,
                 n_output_nodes=600,
                 dropout_probability: float = 0,
                 dimension: Optional[int] = 2,
                 activation_module: Optional[torch.nn.Module] = nn.ReLU,
                 conv_params: Optional[List[List]] = None,
                 nn_layers: Optional[List[int]] = [128],
                 **kwargs):

        assert dimension > 0 and dimension <= 3
        if dimension == 1:
            conv_module = nn.Conv1d
            pool_module = nn.MaxPool1d
        elif dimension == 2:
            conv_module = nn.Conv2d
            pool_module = nn.MaxPool2d
        elif dimension == 3:
            conv_module = nn.Conv3d
            pool_module = nn.MaxPool3d

        if conv_params is None:
            conv_params = [(128, 19, 4), (32, 11, 4), (64, 5, 2), (64, 3, 1),
                           (64, 3, 1)]

        super().__init__(**kwargs)

        self.img_dimension = dimension
        self.dropout_probability = dropout_probability
        self.activation_module = activation_module
        self.n_conv = len(conv_params)
        self.channels = [layer[0] for layer in conv_params]
        self.kernel_sizes = [layer[1] for layer in conv_params]
        self.strides = [layer[2] for layer in conv_params]
        self.n_output_nodes = n_output_nodes

        # Build the CNN
        modules = []
        in_channels = 3  # There are 3 dopants in our initial image
        for _conv_params in conv_params:
            modules.append(
                conv_module(in_channels=in_channels,
                            out_channels=_conv_params[0],
                            kernel_size=_conv_params[1],
                            padding=int((_conv_params[1] - 1) / 2),
                            stride=_conv_params[2]))
            modules.append(activation_module())
            modules.append(pool_module(kernel_size=3, padding=1, stride=2))
            in_channels = _conv_params[0]
        # Flatten before putting the outputinto the FCNN
        modules.append(nn.Flatten())

        for i, n_nodes in enumerate(nn_layers):
            if self.dropout_probability > 0:
                modules.append(nn.Dropout(dropout_probability))
            modules.append(nn.LazyLinear(n_nodes))
            modules.append(activation_module())

        if self.dropout_probability > 0:
            modules.append(nn.Dropout(dropout_probability))
        modules.append(nn.LazyLinear(n_output_nodes))
        modules.append(nn.ReLU())

        self.nn = nn.Sequential(*modules)

        self.save_hyperparameters()

    def forward(self, x, **kwargs):
        x = self.nn(x)

        return x


class DiscreteGraphModel(SpectrumModelBase):

    def __init__(
            self,
            input_channels=3,
            n_output_nodes=400,
            learning_rate: Optional[float] = 1e-5,
            lr_scheduler: Optional[
                Callable[[torch.optim.Optimizer],
                         torch.optim.lr_scheduler._LRScheduler]] = None,
            loss_function: Optional[Callable[[List, List],
                                             float]] = F.mse_loss,
            dropout_probability: float = 0,
            activation_module: Optional[torch.nn.Module] = nn.SiLU,
            mlp_layers=[128, 256],
            mpnn_module: Optional[torch.nn.Module] = pyg_nn.GATv2Conv,
            mpnn_kwargs: Optional[dict] = {'edge_dim': 3},
            mpnn_operation: Optional[str] = 'x, edge_index, edge_attr -> x',
            mpnn_channels=[64, 128],
            readout_operation: Optional[str] = 'attn',
            augment_loss: Optional[bool] = False,
            **kwargs):
        super().__init__(**kwargs)

        self.n_output_nodes = n_output_nodes
        self.learning_rate = learning_rate
        self.dropout_probability = dropout_probability
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
        self.activation_module = activation_module
        self.augment_loss = augment_loss

        # Build the mpnn layers
        mpnn_modules = []
        for i, _ in enumerate(_mlp_channels := [input_channels] +
                              mpnn_channels):
            if i == len(_mlp_channels) - 1:
                break
            mpnn_modules.append((mpnn_module(*_mlp_channels[i:i + 2],
                                             **mpnn_kwargs), mpnn_operation))
            mpnn_modules.append(activation_module(inplace=True))

        # Build the mlp layers
        mlp_modules = []
        for i, _ in enumerate(mlp_sizes := [mpnn_channels[-1] * 2] +
                              mlp_layers + [n_output_nodes]):
            if i == len(mlp_sizes) - 1:
                break
            mlp_modules.append(nn.Dropout())
            mlp_modules.append(nn.Linear(*mlp_sizes[i:i + 2]))
            mlp_modules.append(activation_module(inplace=True))
        # Exclude the last activation, since this will inhibit learning
        mlp_modules = mlp_modules[:-1]

        # Initialize the weights to the linear layers according to Xavier Uniform
        for lin in mlp_modules:
            if isinstance(lin, nn.Linear):
                if activation_module == nn.SiLU:
                    torch.nn.init.xavier_uniform_(
                        lin.weight,
                        gain=1.519)  # 1.519 Seems like a good value for SiLU
                elif activation_module == nn.ReLU:
                    torch.nn.init.kaiming_uniform_(lin.weight,
                                                   nonlinearity='relu')

        if readout_operation.lower() == 'attn':
            # Use attention based Aggregation
            gate_nn = nn.Sequential(
                *[nn.Linear(mpnn_channels[-1], 1),
                  nn.Softmax(-2)])
            out_nn = nn.Linear(mpnn_channels[-1], 2 * mpnn_channels[-1])
            pyg_nn.AttentionalAggregation(gate_nn=gate_nn, nn=out_nn)

            readout = (pyg_nn.AttentionalAggregation(gate_nn=gate_nn,
                                                     nn=out_nn),
                       'x, batch -> x')
        elif readout_operation.lower() == 'set2set':
            # Use the Set2Set aggregation method to pool the
            # graph into a single global feature vector
            readout = (pyg_nn.aggr.Set2Set(mpnn_channels[-1],
                                           processing_steps=10),
                       'x, batch -> x')
        elif readout_operation.lower() == 'sum':
            # Use Sum Aggregation
            readout = (pyg_nn.aggr.SumAggregation(), 'x, batch -> x')
        else:
            # Default to using Mean Aggregation
            readout = (pyg_nn.aggr.MeanAggregation(), 'x, batch -> x')

        # Construct the primary module
        all_modules = mpnn_modules + [readout] + mlp_modules
        self.nn = pyg_nn.Sequential(
            'x, edge_index, edge_attr, edge_weight, batch', all_modules)

        self.save_hyperparameters()

    def forward(self, data):

        x = self.nn(x=data.x[:, :3],
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    edge_weight=data.edge_weight,
                    batch=data.batch)

        if data.batch is None:
            x = x.squeeze()

        return x
