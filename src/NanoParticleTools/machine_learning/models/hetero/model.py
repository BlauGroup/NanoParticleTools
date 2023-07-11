from NanoParticleTools.machine_learning.core import SpectrumModelBase
from NanoParticleTools.machine_learning.modules.layer_interaction import InteractionBlock
from NanoParticleTools.machine_learning.modules.film import FiLMLayer
from NanoParticleTools.machine_learning.modules import NonLinearMLP
from torch_geometric.data.batch import Batch
from torch_geometric.data import Data
from torch.nn import functional as F
from torch import nn
import torch
import torch_geometric.nn as gnn


class DopantInteractionHeteroRepresentationModule(torch.nn.Module):

    def __init__(self,
                 embed_dim: int = 16,
                 n_message_passing: int = 3,
                 self_interaction: bool = False,
                 nsigma=5,
                 **kwargs):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_message_passing = n_message_passing
        self.self_interaction = self_interaction
        self.nsigma = nsigma

        self.dopant_embedder = nn.Embedding(3, embed_dim)
        self.dopant_film_layer = FiLMLayer(1, embed_dim, [16, 16])
        self.dopant_constraint_film_layer = FiLMLayer(2, embed_dim, [16, 16])
        self.dopant_norm = nn.BatchNorm1d(embed_dim)

        self.interaction_embedder = nn.Linear(2, embed_dim)
        self.integrated_interaction = InteractionBlock(nsigma=nsigma)
        self.interaction_norm = nn.BatchNorm1d(nsigma)
        self.interaction_film_layer = FiLMLayer(nsigma, embed_dim, [16, 16])

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
                              add_self_loops=False)
            }
            if self.self_interaction:
                conv_modules[('dopant', 'coupled_to',
                              'self_interaction')] = gnn.GATv2Conv(
                                  embed_dim,
                                  embed_dim,
                                  concat=False,
                                  add_self_loops=False)
                conv_modules[('self_interaction', 'coupled_to',
                              'dopant')] = gnn.GATv2Conv(embed_dim,
                                                         embed_dim,
                                                         concat=False,
                                                         add_self_loops=False)
            self.convs.append(gnn.HeteroConv(conv_modules))

        self.aggregation = gnn.aggr.SumAggregation()

    def forward(self,
                dopant_types,
                dopant_concs,
                dopant_constraint_indices,
                interaction_type_indices,
                interaction_dopant_indices,
                edge_index_dict,
                radii=None,
                batch_dict=None):
        # Embed the dopant types
        dopant_attr = self.dopant_embedder(dopant_types)

        # Use a film layer to condition the dopant embedding on the dopant concentration
        dopant_attr = self.dopant_film_layer(dopant_attr,
                                             dopant_concs.unsqueeze(-1))

        # Condition the dopant node attribute on the size of the dopant constraint
        dopant_node_radii = radii[dopant_constraint_indices]
        dopant_attr = self.dopant_constraint_film_layer(
            dopant_attr, dopant_node_radii)

        # Normalize the dopant node attribute
        dopant_attr = self.dopant_norm(dopant_attr)

        # Embed the interaction nodes, using the pair of dopant types
        interaction_attr = self.interaction_embedder(
            interaction_type_indices.float())
        
        # Index the radii and compute the integrated interaction
        interaction_node_radii = radii[dopant_constraint_indices][
            interaction_dopant_indices].flatten(-2)
        integrated_interaction = self.integrated_interaction(
            *interaction_node_radii.T)
        
        # Multiply the concentration into the integrated_interaction
        interaction_node_conc = dopant_concs[interaction_dopant_indices]
        integrated_interaction = (
            interaction_node_conc[:, 0] *
            interaction_node_conc[:, 1]).unsqueeze(-1) * integrated_interaction
        
        # Normalize the integrated interaction
        integrated_interaction = self.interaction_norm(integrated_interaction)

        # Condition the interaction node attribute on the integrated interaction
        interaction_attr = self.interaction_film_layer(interaction_attr,
                                                       integrated_interaction)

        # Create an dictionary to allow for heterogenous message passing
        intermediate_x_dict = {
            'dopant': dopant_attr,
            'interaction': interaction_attr
        }

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


class DopantInteractionHeteroModel(SpectrumModelBase):

    def __init__(self,
                 embed_dim: int = 16,
                 n_message_passing: int = 3,
                 self_interaction: bool = False,
                 nsigma=5,
                 **kwargs):
        if 'n_input_nodes' in kwargs:
            del kwargs['n_input_nodes']

        super().__init__(n_input_nodes=embed_dim, **kwargs)

        self.embed_dim = embed_dim
        self.n_message_passing = n_message_passing
        self.self_interaction = self_interaction
        self.nsigma = nsigma

        self.representation_module = DopantInteractionHeteroRepresentationModule(
            self.embed_dim, self.n_message_passing, self.self_interaction,
            self.nsigma)

        self.readout = NonLinearMLP(embed_dim, 1, [128], 0.25, nn.SiLU)

    def forward(self,
                dopant_types,
                dopant_concs,
                dopant_constraint_indices,
                interaction_type_indices,
                interaction_dopant_indices,
                edge_index_dict,
                radii=None,
                batch_dict=None):
        representation = self.representation_module(
            dopant_types, dopant_concs, dopant_constraint_indices,
            interaction_type_indices, interaction_dopant_indices,
            edge_index_dict, radii, batch_dict)
        out = self.readout(representation)
        return out

    def get_representation(self, data):
        reps = self.representation_module(
            dopant_types=data['dopant'].types,
            dopant_concs=data['dopant'].x,
            dopant_constraint_indices=data['dopant'].constraint_indices,
            interaction_type_indices=data['interaction'].type_indices,
            interaction_dopant_indices=data['interaction'].dopant_indices,
            edge_index_dict=data.edge_index_dict,
            radii=data.constraint_radii,
            batch_dict=data.batch_dict)
        return reps

    def _evaluate_step(self, data):
        y_hat = self(
            dopant_types=data['dopant'].types,
            dopant_concs=data['dopant'].x,
            dopant_constraint_indices=data['dopant'].constraint_indices,
            interaction_type_indices=data['interaction'].type_indices,
            interaction_dopant_indices=data['interaction'].dopant_indices,
            edge_index_dict=data.edge_index_dict,
            radii=data.constraint_radii,
            batch_dict=data.batch_dict)
        loss = self.loss_function(y_hat, data.log_y)
        return y_hat, loss

    def predict_step(self,
                     batch: Data | Batch,
                     batch_idx: int | None = None) -> torch.Tensor:
        """
        Make a prediction for a batch of data.

        Args:
            batch (_type_): _description_
            batch_idx (int | None, optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: _description_
        """
        y_hat = self(
            dopant_types=batch['dopant'].types,
            dopant_concs=batch['dopant'].x,
            dopant_constraint_indices=batch['dopant'].constraint_indices,
            interaction_type_indices=batch['interaction'].type_indices,
            interaction_dopant_indices=batch['interaction'].dopant_indices,
            edge_index_dict=batch.edge_index_dict,
            radii=batch.constraint_radii,
            batch_dict=batch.batch_dict)
        return y_hat

    def _step(self,
              prefix: str,
              batch: Data | Batch,
              batch_idx: int | None = None,
              log: bool = True):
        y_hat, loss = self._evaluate_step(batch)

        # Determine the batch size
        if batch.batch_dict is not None:
            batch_size = batch.batch_dict["dopant"].size(0)
        else:
            batch_size = 1

        # Log the loss
        metric_dict = {f'{prefix}_loss': loss}
        if prefix != 'train':
            # For the validation and test sets, log additional metrics
            metric_dict[f'{prefix}_mse'] = F.mse_loss(y_hat, batch.log_y)
            metric_dict[f'{prefix}_mae'] = F.l1_loss(y_hat, batch.log_y)
            metric_dict[f'{prefix}_huber'] = F.huber_loss(y_hat, batch.log_y)
            metric_dict[f'{prefix}_hinge'] = F.hinge_embedding_loss(
                y_hat, batch.log_y)
            metric_dict[f'{prefix}_cos_sim'] = F.cosine_similarity(
                y_hat, batch.log_y, 1).mean(0)

        if log:
            self.log_dict(metric_dict, batch_size=batch_size)
        return loss, metric_dict
