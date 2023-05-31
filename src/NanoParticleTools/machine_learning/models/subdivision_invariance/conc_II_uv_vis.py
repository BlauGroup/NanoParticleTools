from NanoParticleTools.machine_learning.models.subdivision_invariance.conc_interaction import (
    SubdivisionInvariantRepresentation)
from typing import Optional
from torch import nn
import torch
from NanoParticleTools.machine_learning.core import SpectrumModelBase
import torch_geometric as pyg
from NanoParticleTools.machine_learning.modules import ParallelModule


class SubdivisionInvariantModel(SpectrumModelBase):

    def __init__(self,
                 n_dopants: int,
                 embed_dim: int,
                 n_output_nodes: int = 8,
                 separate_regression: bool = False,
                 n_sigma: int = 5,
                 sigma: torch.Tensor | None = None,
                 tunable_sigma: bool = True,
                 output_mean: Optional[torch.Tensor] = None,
                 mlp_layers=[128, 256],
                 dropout_probability: float = 0.5,
                 activation_module=nn.SiLU,
                 intermediate_film_dim=None,
                 norm_representation=False,
                 n_message_passing: int = 1,
                 **kwargs):

        if 'n_input_nodes' in kwargs:
            del kwargs['n_input_nodes']

        super().__init__(n_input_nodes=embed_dim, **kwargs)
        self.representation_module = SubdivisionInvariantRepresentation(
            n_dopants=n_dopants,
            embed_dim=embed_dim,
            sigma=sigma,
            n_sigma=n_sigma,
            tunable_sigma=tunable_sigma,
            intermediate_film_dim=intermediate_film_dim,
            n_message_passing=n_message_passing)

        self.n_dopants = n_dopants
        self.embed_dim = embed_dim
        self.n_output_nodes = n_output_nodes
        self.norm_representation = norm_representation
        self.dropout_probability = dropout_probability

        if separate_regression:
            module_list = []
            for _ in range(n_output_nodes):
                # Build the mlp layers
                mlp_modules = []
                mlp_sizes = [embed_dim] + mlp_layers + [1]
                for i, _ in enumerate(mlp_sizes):
                    if i == len(mlp_sizes) - 1:
                        break
                    mlp_modules.append(nn.Dropout(dropout_probability))
                    mlp_modules.append(nn.Linear(*mlp_sizes[i:i + 2]))
                    mlp_modules.append(activation_module(inplace=True))
                # Exclude the last activation, since this will inhibit learning
                mlp_modules = mlp_modules[:-1]
                module_list.append(torch.nn.Sequential(*mlp_modules))
            self.regressor = ParallelModule(*module_list)
        else:
            # Build the mlp layers
            mlp_modules = []
            mlp_sizes = [embed_dim] + mlp_layers + [n_output_nodes]
            for i, _ in enumerate(mlp_sizes):
                if i == len(mlp_sizes) - 1:
                    break
                mlp_modules.append(nn.Dropout(dropout_probability))
                mlp_modules.append(nn.Linear(*mlp_sizes[i:i + 2]))
                mlp_modules.append(activation_module(inplace=True))
            # Exclude the last activation, since this will inhibit learning
            mlp_modules = mlp_modules[:-1]
            self.regressor = torch.nn.Sequential(*mlp_modules)

        self.save_hyperparameters()

    def forward(self,
                types,
                x_dopant,
                node_dopant_index,
                radii,
                x_layer_idx,
                edge_index,
                batch=None,
                **kwargs):
        # Generate the representation of the nanoparticle
        rep = self.representation_module(types, x_dopant, node_dopant_index,
                                         radii, x_layer_idx, edge_index, batch)

        # Make the spectrum prediction
        out = self.regressor(rep)

        return out
