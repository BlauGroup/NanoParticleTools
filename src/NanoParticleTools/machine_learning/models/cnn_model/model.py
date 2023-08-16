import torch
from torch import nn
from typing import Optional, List
from NanoParticleTools.machine_learning.core import SpectrumModelBase
from NanoParticleTools.machine_learning.modules.core import LazyNonLinearMLP


class CNNModel(SpectrumModelBase):

    def __init__(self,
                 n_output_nodes=600,
                 dropout_probability: float = 0,
                 dimension: Optional[int] = 1,
                 activation_module: Optional[torch.nn.Module] = nn.ReLU,
                 conv_params: Optional[List[List]] = None,
                 readout_layers: Optional[List[int]] = None,
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

        if readout_layers is None:
            readout_layers = [128]
        self.readout_layers = readout_layers

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

        self.representation_module = nn.Sequential(*modules)

        self.readout = LazyNonLinearMLP(n_output_nodes, self.readout_layers,
                                        0.25, self.activation_module)
        self.save_hyperparameters()

    def forward(self, x, **kwargs):
        representation = self.representation_module(x)

        out = self.readout(representation)
        return out

    def get_representation(self, data):
        """
        Get the representation latent vector for a batch of data
        """
        reps = self.representation_module(data.x)
        return reps
