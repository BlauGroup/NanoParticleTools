from NanoParticleTools.machine_learning.core import SpectrumModelBase
from NanoParticleTools.machine_learning.modules.layer_interaction import InteractionBlock
from NanoParticleTools.machine_learning.modules.film import FiLMLayer
from NanoParticleTools.machine_learning.modules import NonLinearMLP
from torch_geometric.data.batch import Batch
from torch_geometric.data import HeteroData
from torch.nn import functional as F
from torch import nn
import torch
import torch_geometric.nn as gnn
import warnings
from typing import Dict, List


class HeteroDCVRepresentationModule(torch.nn.Module):

    def __init__(self,
                 embed_dim: int = 16,
                 n_message_passing: int = 3,
                 nsigma: int = 5,
                 interaction_embedding: bool = True,
                 conc_eps: float = 0.01,
                 geometry_film_layers: List[int] = [16, 16],
                 n_dopants: int = 3,
                 **kwargs):
        """
        Args:
            embed_dim: _description_.
            n_message_passing: _description_.
            nsigma: _description_.
            volume_normalization: _description_.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.n_message_passing = n_message_passing
        self.nsigma = nsigma
        self.conc_eps = conc_eps
        self.interaction_embedding = interaction_embedding

        self.dopant_embedder = nn.Embedding(n_dopants, embed_dim)
        self.dopant_film_layer = FiLMLayer(1, embed_dim, [16, 16])

        self.dopant_constraint_film_layer = FiLMLayer(2, embed_dim, [16, 16])
        self.dopant_norm = nn.BatchNorm1d(embed_dim)

        if self.interaction_embedding:
            self.interaction_embedder = nn.Embedding(n_dopants**2, embed_dim)
            self.intraaction_embedder = nn.Embedding(n_dopants**2, embed_dim)
        else:
            self.interaction_embedder = nn.Linear(2, embed_dim)
            self.intraaction_embedder = nn.Linear(2, embed_dim)
        self.integrated_interaction = InteractionBlock(nsigma=nsigma)
        self.interaction_norm = nn.BatchNorm1d(nsigma)
        self.intraaction_norm = nn.BatchNorm1d(nsigma)
        self.interaction_film_layer = FiLMLayer(nsigma, embed_dim, [16, 16])
        self.intraaction_film_layer = FiLMLayer(nsigma, embed_dim, [16, 16])

        self.convs = nn.ModuleList()
        for _ in range(n_message_passing):
            conv_modules = {
                ('dopant', 'coupled_to', 'interaction'):
                gnn.GATv2Conv(embed_dim,
                              embed_dim,
                              concat=False,
                              add_self_loops=False),
                ('interaction', 'coupled_to', 'dopant'):
                gnn.GATv2Conv(embed_dim,
                              embed_dim,
                              concat=False,
                              add_self_loops=False),
                ('dopant', 'coupled_to', 'intraaction'):
                gnn.GATv2Conv(embed_dim,
                              embed_dim,
                              concat=False,
                              add_self_loops=False),
                ('intraaction', 'coupled_to', 'dopant'):
                gnn.GATv2Conv(embed_dim,
                              embed_dim,
                              concat=False,
                              add_self_loops=False)
            }
            self.convs.append(gnn.HeteroConv(conv_modules))

        self.aggregation = gnn.aggr.SumAggregation()

    def forward(self,
                dopant_types,
                dopant_concs,
                dopant_constraint_indices,
                interaction_type_indices,
                interaction_types,
                interaction_dopant_indices,
                intraaction_type_indices,
                intraaction_types,
                intraaction_dopant_indices,
                edge_index_dict,
                radii,
                constraint_radii_idx,
                batch_dict=None):
        # Index the radii
        _radii = radii[constraint_radii_idx]

        # Compute the volumes of the constraints
        volume = 4 / 3 * torch.pi * _radii**3
        volume = volume[:, 1] - volume[:, 0]

        # Embed the dopant types
        dopant_attr = self.dopant_embedder(dopant_types)

        # Use a film layer to condition the dopant embedding on the dopant concentration
        conc_attr = dopant_concs.unsqueeze(-1)
        # conc_attr = self.conc_norm(conc_attr)
        dopant_attr = self.dopant_film_layer(dopant_attr, conc_attr)

        # Condition the dopant node attribute on the size of the dopant constraint
        geometric_features = _radii[dopant_constraint_indices]
        # geometric_features = self.geometry_norm(geometric_features)

        dopant_attr = self.dopant_constraint_film_layer(
            dopant_attr, geometric_features)

        # Normalize the dopant node attribute
        dopant_attr = self.dopant_norm(dopant_attr)

        # Create an dictionary to allow for heterogenous message passing
        intermediate_x_dict = {
            'dopant': dopant_attr,
        }

        # Embed the interaction nodes, using the pair of dopant types
        if self.interaction_embedding:
            interaction_attr = self.interaction_embedder(interaction_types)
        else:
            interaction_attr = self.interaction_embedder(
                interaction_type_indices.float())

        # Index the radii and compute the integrated interaction
        interaction_node_radii = _radii[dopant_constraint_indices][
            interaction_dopant_indices].flatten(-2)
        integrated_interaction = self.integrated_interaction(
            *interaction_node_radii.T)

        # Multiply the concentration into the integrated_interaction
        interaction_node_conc = dopant_concs[interaction_dopant_indices]
        conc_factor = (interaction_node_conc[:, 0] *
                       interaction_node_conc[:, 1]).unsqueeze(-1)
        integrated_interaction = conc_factor * integrated_interaction

        # Normalize the integrated interaction
        # Using a batch norm
        integrated_interaction = self.interaction_norm(integrated_interaction)

        # Condition the interaction node attribute on the integrated interaction
        interaction_attr = self.interaction_film_layer(interaction_attr,
                                                       integrated_interaction)
        intermediate_x_dict['interaction'] = interaction_attr

        # Repeat the same process for the intraaction
        # Embed the interaction nodes, using the pair of dopant types
        if self.interaction_embedding:
            intraaction_attr = self.intraaction_embedder(intraaction_types)
        else:
            intraaction_attr = self.intraaction_embedder(
                intraaction_type_indices.float())

        # Index the radii and compute the integrated intraaction
        intraaction_node_radii = _radii[dopant_constraint_indices][
            intraaction_dopant_indices].flatten(-2)
        integrated_intraaction = self.integrated_interaction(
            *intraaction_node_radii.T)

        # Multiply the concentration into the integrated_intraaction
        intraaction_node_conc = dopant_concs[intraaction_dopant_indices]
        conc_factor = (intraaction_node_conc[:, 0] *
                       intraaction_node_conc[:, 1]).unsqueeze(-1)
        integrated_intraaction = conc_factor * integrated_intraaction

        # Normalize the integrated intraaction
        # Using a batch norm
        integrated_intraaction = self.intraaction_norm(integrated_intraaction)

        # Condition the intraaction node attribute on the integrated intraaction
        intraaction_attr = self.intraaction_film_layer(intraaction_attr,
                                                       integrated_intraaction)
        intermediate_x_dict['intraaction'] = intraaction_attr

        # Apply the message passing operator(s)
        for conv in self.convs:
            intermediate_x_dict = conv(intermediate_x_dict, edge_index_dict)
            intermediate_x_dict = {
                k: nn.functional.silu(v)
                for k, v in intermediate_x_dict.items()
            }

        if batch_dict and 'dopant' in batch_dict:
            out = self.aggregation(intermediate_x_dict['dopant'],
                                   batch_dict['dopant'])
        else:
            out = self.aggregation(intermediate_x_dict['dopant'])

        return out


class HeteroDCVModel(SpectrumModelBase):

    def __init__(self,
                 n_dopants: int = 3,
                 embed_dim: int = 16,
                 n_message_passing: int = 3,
                 nsigma=5,
                 readout_layers: List[int] = [128],
                 **kwargs):
        """
        Args:
            n_dopants: The number of dopants in the system.
            embed_dim: The dimension of the embedding/latent space.
            n_message_passing: The number of heterogeneous message passing steps.
            nsigma: The number of std deviations to use in the gaussian of the integrated
                interaction. Also corresponds to the number of channels output by the
                integrated interaction.
            readout_layers: The number of layers in the readout MLP.

        Inherited Args:
            l2_regularization_weight: The weight of the L2 regularization term in the loss function.
                This is passed to the torch optimizer
            optimizer_type: The type of optimizer to use. options are 'sgd', 'adam', and 'amsgrad'.
                if 'amsgrad' is selected, the pytorch adam optimizer is used with the `amsgrad=True`
            learning_rate: The default learning rate for model training. The actual learning rate
                used may be different depending on the actions of the learning rate scheduler
            lr_scheduler: The learning rate scheduler class to use.
            lr_scheduler_kwargs: The kwargs passed to the learning rate scheduler on initialization.
            loss_function: The loss function to use for backpropagation in training.
                MAE, MSE, and Cosine Similarity will be logged anyways.
            additional_metadata: Additional metadata which will be logged with the model to
                wandb.
        """
        if 'n_input_nodes' in kwargs:
            warnings.warn(
                'Cannot override n_input_nodes for this model. It is inferred from'
                'the embed_dim.')
            del kwargs['n_input_nodes']

        super().__init__(n_input_nodes=embed_dim, **kwargs)

        self.embed_dim = embed_dim
        self.n_message_passing = n_message_passing
        self.nsigma = nsigma
        self.readout_layers = readout_layers

        self.representation_module = HeteroDCVRepresentationModule(
            self.embed_dim,
            self.n_message_passing,
            self.nsigma,
            n_dopants=n_dopants,
            **kwargs)

        self.readout = NonLinearMLP(embed_dim, 1, self.readout_layers, 0.25,
                                    nn.SiLU)

        self.save_hyperparameters()

    def forward(self,
                dopant_types,
                dopant_concs,
                dopant_constraint_indices,
                interaction_type_indices,
                interaction_types,
                interaction_dopant_indices,
                intraaction_type_indices,
                intraaction_types,
                intraaction_dopant_indices,
                edge_index_dict,
                radii,
                constraint_radii_idx,
                batch_dict=None):
        representation = self.representation_module(
            dopant_types, dopant_concs, dopant_constraint_indices,
            interaction_type_indices, interaction_types,
            interaction_dopant_indices, intraaction_type_indices,
            intraaction_types, intraaction_dopant_indices, edge_index_dict,
            radii, constraint_radii_idx, batch_dict)
        out = self.readout(representation)
        return out

    def get_inputs(self, data: HeteroData) -> Dict:

        input_dict = {
            'dopant_types': data['dopant'].types,
            'dopant_concs': data['dopant'].x,
            'dopant_constraint_indices': data['dopant'].constraint_indices,
            'interaction_type_indices': data['interaction'].type_indices,
            'interaction_types': data['interaction'].types,
            'interaction_dopant_indices': data['interaction'].dopant_indices,
            'intraaction_type_indices': data['intraaction'].type_indices,
            'intraaction_types': data['intraaction'].types,
            'intraaction_dopant_indices': data['intraaction'].dopant_indices,
            'edge_index_dict': data.edge_index_dict,
            'radii': data.radii,
            'constraint_radii_idx': data.constraint_radii_idx,
            'batch_dict': data.batch_dict
        }
        return input_dict

    def get_representation(self, data):
        reps = self.representation_module(**self.get_inputs(data))
        return reps

    def predict_step(self,
                     batch: HeteroData | Batch,
                     batch_idx: int | None = None) -> torch.Tensor:
        """
        Make a prediction for a batch of data.

        Args:
            batch (_type_): _description_
            batch_idx (int | None, optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: _description_
        """
        y_hat = self(**self.get_inputs(batch))
        return y_hat

    def evaluate_step(self, data):
        """
        A single forward pass of the data and loss calculation.

        Args:
            batch: A single data point or a collated batch of data points.

        Returns:
            y_hat: The predicted value(s) for the batch
            loss: The loss for the batch
        """
        y_hat = self(**self.get_inputs(data))

        loss = self.loss_function(y_hat, data.log_y)
        return y_hat, loss

    def get_batch_size(self, batch):
        if batch.batch_dict is not None:
            return batch.batch_dict["dopant"].size(0)
        else:
            return 1
