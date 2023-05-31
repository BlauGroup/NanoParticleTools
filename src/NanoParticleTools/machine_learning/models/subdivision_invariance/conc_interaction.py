from typing import Optional
from torch import nn
import torch
from NanoParticleTools.machine_learning.modules.core import BatchScaling
from NanoParticleTools.machine_learning.modules.film import FiLMLayer
from NanoParticleTools.machine_learning.modules.layer_interaction import InteractionBlock
from NanoParticleTools.machine_learning.core.model import SpectrumModelBase
import torch_geometric as pyg


class SubdivisionInvariantRepresentation(nn.Module):

    def __init__(self,
                 n_dopants: int = 3,
                 embed_dim: int = 7,
                 sigma: torch.Tensor | None = None,
                 n_sigma: int = 5,
                 tunable_sigma: bool = True,
                 intermediate_film_dim: int | None = None,
                 n_message_passing: int = 1,
                 norm_representation: bool = True,
                 norm_interaction: bool = False,
                 batch_norm: bool = False):
        super().__init__()
        self.embedder = nn.Embedding(n_dopants**2, embed_dim)
        self.film_layer = FiLMLayer(2, embed_dim, intermediate_film_dim)
        self.integrated_interaction = InteractionBlock(
            sigma=sigma,
            nsigma=n_sigma,
            tunable_sigma=tunable_sigma,
            tanh_approx=False)
        self.interaction_bilin = nn.Bilinear(n_sigma,
                                             embed_dim,
                                             embed_dim,
                                             bias=False)

        self.message_layer = torch.nn.ModuleList()
        for i in range(n_message_passing):
            self.message_layer.append(
                pyg.nn.GATv2Conv(embed_dim,
                                 embed_dim,
                                 heads=1,
                                 concat=False,
                                 dropout=0.1))
        
        self.batch_norm_layers = torch.nn.ModuleList()
        self.batch_norm = batch_norm
        for i in range(n_message_passing):
            if batch_norm:
                self.batch_norm_layers.append(pyg.nn.norm.BatchNorm(embed_dim,
                                                                    affine=True,
                                                                    allow_single_element=True))
            else:
                self.batch_norm_layers.append(nn.Identity())
        self.readout = pyg.nn.aggr.SumAggregation()
        self.norm_representation = norm_representation

        self.norm_interaction = norm_interaction
        if norm_interaction:
            self.interaction_norm = pyg.nn.norm.BatchNorm(n_sigma,
                                                          affine=False,
                                                          allow_single_element=True)
        else:
            # We use scaling, since the typical function for standardizing
            # data $\frac{x-\hat{x}}{\sigma}$ is not additive.
            self.interaction_norm = BatchScaling(n_sigma)

    def forward(self,
                types,
                x_dopant,
                node_dopant_index,
                radii,
                x_layer_idx,
                edge_index,
                batch=None,
                **kwargs):
        # Embedding
        embedding = self.embedder(types)

        # Condition the embedding on the composition
        x_embedding = self.film_layer(embedding, x_dopant[node_dopant_index])

        radii = radii[x_layer_idx][node_dopant_index].double()
        # Cast back to a float, since we don't want to keep the whole model in double precision.
        integrated_interaction = self.integrated_interaction(radii[:, 0, 0], radii[:, 0, 1],
                                                             radii[:, 1, 0], radii[:, 1, 1]).float()
        # Multiply by the concentrations
        integrated_interaction = integrated_interaction * x_dopant[node_dopant_index][..., 0].unsqueeze(-1) * x_dopant[node_dopant_index][..., 1].unsqueeze(-1)
        integrated_interaction = torch.nn.functional.relu(integrated_interaction)

        # Scale the integrated interaction values according to the mean Interaction values.
        I_scaled = self.interaction_norm(integrated_interaction)

        # Apply the integrated interaction to the edges using a bilinear operation
        node_attr = self.interaction_bilin(I_scaled, x_embedding)

        # Message Passing
        for message_layer, norm_layer in zip(self.message_layer, self.batch_norm_layers):
            node_attr = message_layer(node_attr, edge_index)
            node_attr = norm_layer(node_attr)

        # Readout
        out = self.readout(node_attr, batch)

        # Normalize the representation, so that the representation space is bounded
        if self.norm_representation:
            out = torch.nn.functional.normalize(out, dim=1)
        return out


class SubdivisionInvariantModel(SpectrumModelBase):

    def __init__(self,
                 n_dopants: int,
                 embed_dim: int,
                 n_output_nodes: int = 600,
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

        if output_mean is not None:
            self.register_buffer("output_mean", output_mean)
        else:
            self.output_mean = output_mean

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
        self.nn = torch.nn.Sequential(*mlp_modules)

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
        out = self.nn(rep)

        if self.output_mean is not None:
            return out + self.output_mean
        else:
            return out
