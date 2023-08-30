from NanoParticleTools.machine_learning.core import SpectrumModelBase
from NanoParticleTools.machine_learning.modules import InteractionConv
from torch import nn
import torch

from torch_geometric import nn as pyg_nn


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
        self.conv1 = InteractionConv(input_dim=embed_dim,
                                     output_dim=embed_dim,
                                     cat_embedding=True)
        self.conv2 = InteractionConv(input_dim=2 * embed_dim,
                                     output_dim=embed_dim,
                                     cat_embedding=True)

        # Build the mlp layers
        if readout_operation.lower() == 'set2set':
            mlp_sizes = [2 * n_dopants * 2 * embed_dim
                         ] + mlp_layers + [n_output_nodes]
        else:
            mlp_sizes = [n_dopants * 2 * embed_dim
                         ] + mlp_layers + [n_output_nodes]

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
                *[nn.Linear(n_dopants * 2 * embed_dim, 1),
                  nn.Softmax(-2)])
            out_nn = nn.Linear(n_dopants * 2 * embed_dim,
                               n_dopants * 2 * embed_dim)
            pyg_nn.AttentionalAggregation(gate_nn=gate_nn, nn=out_nn)

            readout = pyg_nn.AttentionalAggregation(gate_nn=gate_nn, nn=out_nn)
        elif readout_operation.lower() == 'set2set':
            # Use the Set2Set aggregation method to pool the graph
            # into a single global feature vector
            readout = pyg_nn.aggr.Set2Set(n_dopants * 2 * embed_dim,
                                          processing_steps=7)
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

    def forward(self, x, edge_index, edge_attr, batch, **kwargs):
        embedding = self.embeder(
            torch.arange(0, self.n_dopants, device=x.device).expand(x.shape))
        out = self.conv1(embedding, x, edge_index, edge_attr)
        out = torch.concat((embedding, out), dim=-1)
        out = self.conv2(out, x, edge_index, edge_attr)
        out = torch.concat((embedding, out), dim=-1)

        # Flatten the embeddings
        out = out.reshape(out.size(0), -1)

        #         # Node level readout
        #         out = self.nn(out)

        # Global Readout
        out = self.readout(out, batch)
        out = self.nn(out)

        return out
