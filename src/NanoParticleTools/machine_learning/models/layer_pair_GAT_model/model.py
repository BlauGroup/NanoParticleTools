import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric import nn as pyg_nn
import pytorch_lightning as pl

from ...core.model import SpectrumModelBase
from ..mlp_model.model import MLPSpectrumModel
from typing import Optional, Callable
from torch_scatter.scatter import scatter

NODE_EMBEDDING_INDEX = 0
NODE_DISTANCE_INDEX = 1
NODE_COMPOSITION_INDEX = 2
NODE_VOLUME_INDEX = 4


class GATSpectrumModel(MLPSpectrumModel):

    def __init__(
            self,
            in_feature_dim: int,
            out_feature_dim: Optional[int] = None,
            nonlinear_node_info: Optional[bool] = True,
            concatenate_embedding: Optional[bool] = False,
            #  mlp_embedding: Optional[List[int]] = [256],
            mlp_embedding: Optional[bool] = True,
            second_mlp_embedding: Optional[bool] = True,
            weight_pair_sum: Optional[bool] = False,
            second_mpnn: Optional[bool] = True,
            heads=4,
            gnn_dropout_probability: Optional[float] = 0.6,
            gnn_activation: Optional[Callable] = F.leaky_relu,
            sum_attention_output: Optional[bool] = False,
            embedding_dictionary_size: Optional[int] = 9,
            resnet: Optional[bool] = False,
            **kwargs):
        """
        There are a few ways in which we can add composition and volume information to the model:
        - Option 1: Multiply by composition
        - Option 2: Concatenate the additional information
        - Option 3: Concatenate, then MLP
        - Option 3: MLP the additional information, then concatenate
        - Option 4: MLP the additional information, then concatenate, then MLP
        To achieve these, we can optionally apply 3 operations
        1) Optionally applying a MLP to the info
        2) Concatenating or Multiplying
        3) Optionally applying an MLP to the embedding

        :param nonlinear_node_info: whether or not to apply operation 1
        :param concatenate: whether to concatenate or multiply (default) with the embedding
        :param mlp_embedding: whether or not to apply operation 2

        """
        if out_feature_dim is None:
            out_feature_dim = in_feature_dim
        kwargs['n_input_nodes'] = out_feature_dim
        super().__init__(**kwargs)

        embedding_dim = in_feature_dim

        if nonlinear_node_info:
            self.node_mlp = nn.Sequential(nn.Linear(5, 128), nn.Linear(128, 5))

        if concatenate_embedding:
            embedding_dim = embedding_dim - 5

        if mlp_embedding:
            self.node_embedding_mlp = nn.Sequential(
                nn.Linear(in_feature_dim, 256), nn.Linear(256, in_feature_dim))

        if second_mlp_embedding:
            self.second_node_embedding_mlp = nn.Sequential(
                nn.Linear(in_feature_dim, 256), nn.Linear(256, in_feature_dim))

        self.embedder = nn.Embedding(embedding_dictionary_size, embedding_dim)
        if second_mpnn:
            self.conv1 = pyg_nn.GATv2Conv(in_feature_dim,
                                          in_feature_dim,
                                          add_self_loops=False,
                                          heads=heads)
            self.conv2 = pyg_nn.GATv2Conv(in_feature_dim * heads,
                                          out_feature_dim,
                                          add_self_loops=False,
                                          heads=heads,
                                          concat=False)
        else:
            self.conv1 = pyg_nn.GATv2Conv(in_feature_dim,
                                          out_feature_dim,
                                          add_self_loops=False,
                                          heads=heads,
                                          concat=False)

        if weight_pair_sum:
            self.pair_weight = nn.Embedding(embedding_dictionary_size, 1)

        self.in_feature_dim = in_feature_dim
        self.out_feature_dim = out_feature_dim
        self.nonlinear_node_info = nonlinear_node_info
        self.concatenate_embedding = concatenate_embedding
        self.mlp_embedding = mlp_embedding
        self.second_mlp_embedding = second_mlp_embedding
        self.weight_pair_sum = weight_pair_sum
        self.second_mpnn = second_mpnn
        self.resnet = resnet
        self.heads = heads
        self.gnn_dropout_probability = gnn_dropout_probability
        self.gnn_activation = gnn_activation
        self.sum_attention_output = sum_attention_output
        self.embedding_dictionary_size = embedding_dictionary_size

    def forward(self, data):

        pair_identity = data.x[:, 0].long()
        x = self.embedder(pair_identity)
        additional_info = data.x[:, 1:]

        # Add composition and volume information to the embedding
        # Step 1: MLP composition or not
        if self.nonlinear_node_info:
            additional_info = self.node_mlp(additional_info)

        # Step 2: Concatenate or Multiply
        if self.concatenate_embedding:
            x = torch.concat([x, additional_info], dim=1)
        else:
            for i in range(0, 5):
                x = x * additional_info[:, i].expand(self.in_feature_dim,
                                                     -1).transpose(0, 1)
        # Step 3: MLP or not
        if self.mlp_embedding:
            x = self.node_embedding_mlp(x)

        if self.second_mlp_embedding:
            x = self.second_node_embedding_mlp(x)

        # Apply the first layer of the MPNN
        _x = self.conv1(x, data.edge_index)
        _x = self.gnn_activation(_x)
        if self.resnet:
            x = x.expand(self.heads, *x.shape).moveaxis(0, -2).reshape(
                *x.shape[:-1], -1) + _x
        else:
            x = _x

        if self.second_mpnn:
            # Apply the second layer of the MPNN
            _x = F.dropout(x,
                           self.gnn_dropout_probability,
                           training=self.training)
            _x = self.conv2(_x, data.edge_index)
            if self.resnet:
                x = x.reshape(*x.shape[:-1], 4, -1).mean(-2) + _x
            else:
                x = _x

        if self.sum_attention_output:
            if data.batch is not None:
                x = scatter(x, data.batch.unsqueeze(1), dim=0, reduce='sum')
            else:
                x = torch.sum(x, dim=0)
            output = self.nn(x)
        else:
            x = self.nn(x)

            # Add a weight to the pairwise sum
            if self.weight_pair_sum:
                x = torch.einsum(
                    '...ij, ...i -> ...ij', x,
                    self.pair_weight(pair_identity).squeeze(dim=-1))

            if data.batch is not None:
                output = scatter(x,
                                 data.batch.unsqueeze(1),
                                 dim=0,
                                 reduce='sum')
            else:
                output = torch.sum(x, dim=0)
        return output
