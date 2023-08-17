import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric import nn as pyg_nn
import pytorch_lightning as pl

from NanoParticleTools.machine_learning.core import SpectrumModelBase
from NanoParticleTools.machine_learning.models.mlp_model.model import MLPSpectrumModel
from typing import Optional, Callable
from torch_scatter.scatter import scatter


class GATSpectrumModel(SpectrumModelBase):

    def __init__(self,
                 n_dopants: int,
                 embed_dim: int,
                 n_output_nodes=600,
                 mlp_layers=[128, 256],
                 dropout_probability: float = 0,
                 activation_module=nn.SiLU,
                 readout_operation='mean',
                 **kwargs):

        if 'n_input_nodes' in kwargs:
            del kwargs['n_input_nodes']

        super().__init__(n_input_nodes=2 * embed_dim, **kwargs)

        self.n_dopants = n_dopants
        self.embed_dim = embed_dim
        self.n_output_nodes = n_output_nodes
        self.embeder = nn.Embedding(n_dopants, embed_dim)
        self.conv1 = pyg_nn.GATv2Conv(embed_dim + 1,
                                      embed_dim,
                                      edge_dim=4,
                                      add_self_loops=False,
                                      heads=1,
                                      concat=False)
        self.conv2 = pyg_nn.GATv2Conv(2 * embed_dim,
                                      embed_dim,
                                      edge_dim=4,
                                      add_self_loops=False,
                                      heads=1,
                                      concat=False)

        # Build the mlp layers
        if readout_operation.lower() == 'set2set':
            mlp_sizes = [2 * 2 * embed_dim] + mlp_layers + [n_output_nodes]
        else:
            mlp_sizes = [2 * embed_dim] + mlp_layers + [n_output_nodes]

        mlp_modules = []
        for i, _ in enumerate(mlp_sizes):
            if i == len(mlp_sizes) - 1:
                break
            mlp_modules.append(nn.Dropout())
            mlp_modules.append(nn.Linear(*mlp_sizes[i:i + 2]))
            mlp_modules.append(activation_module(inplace=True))
        # Exclude the last activation, since this will inhibit learning
        mlp_modules = mlp_modules[:-1]
        self.nn = torch.nn.Sequential(*mlp_modules)

        if readout_operation.lower() == 'attn':
            # Use attention based Aggregation
            gate_nn = nn.Sequential(
                *[nn.Linear(2 * embed_dim, 1),
                  nn.Softmax(-2)])
            out_nn = nn.Linear(2 * embed_dim, 2 * embed_dim)
            pyg_nn.AttentionalAggregation(gate_nn=gate_nn, nn=out_nn)

            readout = pyg_nn.AttentionalAggregation(gate_nn=gate_nn, nn=out_nn)
        elif readout_operation.lower() == 'set2set':
            # Use the Set2Set aggregation method to pool the graph
            # into a single global feature vector
            readout = pyg_nn.aggr.Set2Set(2 * embed_dim, processing_steps=7)
        elif readout_operation.lower() == 'sum':
            # Use Sum Aggregation
            readout = pyg_nn.aggr.SumAggregation()
        elif readout_operation.lower() == 'mean':
            # Use Mean Aggregation
            readout = pyg_nn.aggr.MeanAggregation()
        else:
            # TODO: Default to node prediction, then sum
            raise ValueError("readout not specified")

        self.readout = readout

        self.save_hyperparameters()

    def forward(self, x, types, edge_index, edge_attr, batch, **kwargs):
        embedding = self.embeder(types)
        out = torch.concat((embedding, x.unsqueeze(-1)), dim=-1)

        out = self.conv1(out, edge_index, edge_attr)
        out = torch.concat((embedding, out), dim=-1)
        out = self.conv2(out, edge_index, edge_attr)
        out = torch.concat((embedding, out), dim=-1)

        # Flatten the embeddings
        # out = out.reshape(out.size(0), -1)

        #         # Node level readout
        #         out = self.nn(out)

        # Global Readout
        out = self.readout(out, batch)
        out = self.nn(out)

        return out.squeeze()
        # return out


class GATEdgeSpectrumModel(MLPSpectrumModel):

    def __init__(self,
                 in_feature_dim: int,
                 edge_dim: int,
                 out_feature_dim: Optional[int] = 8,
                 nonlinear_node_info: Optional[bool] = True,
                 concatenate_embedding: Optional[bool] = False,
                 mlp_embedding: Optional[bool] = True,
                 heads=4,
                 gnn_dropout_probability: Optional[float] = 0.6,
                 gnn_activation: Optional[Callable] = F.leaky_relu,
                 sum_attention_output: Optional[bool] = False,
                 embedding_dictionary_size: Optional[int] = 3,
                 **kwargs):

        if 'n_input_nodes' in kwargs:
            del kwargs['n_input_nodes']

        super().__init__(n_input_nodes=in_feature_dim, **kwargs)

        embedding_dim = in_feature_dim

        if nonlinear_node_info:
            self.node_mlp = nn.Sequential(nn.Linear(6, 128), nn.Linear(128, 6))

        if concatenate_embedding:
            embedding_dim = embedding_dim - 6

        if mlp_embedding:
            self.node_embedding_mlp = nn.Sequential(
                nn.Linear(in_feature_dim, 256), nn.Linear(256, in_feature_dim))

        self.embedder = nn.Embedding(embedding_dictionary_size, embedding_dim)
        self.conv1 = pyg_nn.GATv2Conv(in_feature_dim,
                                      in_feature_dim,
                                      edge_dim=edge_dim,
                                      add_self_loops=False,
                                      heads=heads,
                                      concat=False)
        self.conv2 = pyg_nn.GATv2Conv(in_feature_dim,
                                      in_feature_dim,
                                      edge_dim=edge_dim,
                                      add_self_loops=False,
                                      heads=heads,
                                      concat=False)

        self.in_feature_dim = in_feature_dim
        self.edge_dim = edge_dim
        self.out_feature_dim = out_feature_dim
        self.nonlinear_node_info = nonlinear_node_info
        self.concatenate_embedding = concatenate_embedding
        self.mlp_embedding = mlp_embedding
        self.heads = heads
        self.gnn_dropout_probability = gnn_dropout_probability
        self.gnn_activation = gnn_activation
        self.sum_attention_output = sum_attention_output
        self.embedding_dictionary_size = embedding_dictionary_size

    def forward(self, data):
        """
        Steps:
            1) Use an embedding to represent the features
            2) Construct the feature vector consisting of
                {(inner, middle, and outer radius), (embedding), (learned_features)}
            2) Apply non-linearity to the volume and composition
            3) Apply non-linearity to the distances
        """

        edge_index, edge_attr = data.edge_index, data.edge_attr

        # Embedding the atom identity
        atom_identity = data.x[:, 0].long()
        x = self.embedder(atom_identity)
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

        # Apply the first layer of the MPNN
        x = self.conv1(x, data.edge_index, edge_attr=edge_attr)
        x = self.gnn_activation(x)

        # Apply the second layer of the MPNN
        x = F.dropout(x, self.gnn_dropout_probability, training=self.training)
        x = self.conv2(x, data.edge_index, edge_attr=edge_attr)

        # Readout
        if self.sum_attention_output:
            if data.batch is not None:
                x = scatter(x, data.batch.unsqueeze(1), dim=0, reduce='sum')
            else:
                x = torch.sum(x, dim=0)
            output = self.nn(x)
        else:
            x = self.nn(x)

            if data.batch is not None:
                output = scatter(x,
                                 data.batch.unsqueeze(1),
                                 dim=0,
                                 reduce='sum')
            else:
                output = torch.sum(x, dim=0)
        return output