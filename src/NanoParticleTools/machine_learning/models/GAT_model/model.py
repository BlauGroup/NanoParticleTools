import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric import nn as pyg_nn
import pytorch_lightning as pl

from .._model import SpectrumModelBase
from typing import Optional, Callable
from torch_scatter.scatter import scatter

class GATSpectrumModel(SpectrumModelBase):
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
        super().__init__(n_input_nodes = in_feature_dim, **kwargs)

        embedding_dim = in_feature_dim

        if nonlinear_node_info:
            self.node_mlp = nn.Sequential(nn.Linear(6, 128),
                                          nn.Linear(128, 6))
        
        if concatenate_embedding:
            embedding_dim = embedding_dim - 6
            
        if mlp_embedding:
            self.node_embedding_mlp = nn.Sequential(nn.Linear(in_feature_dim, 256),
                                                    nn.Linear(256, in_feature_dim))
        
        self.embedder = nn.Embedding(embedding_dictionary_size, embedding_dim)
        self.conv1 = pyg_nn.GATv2Conv(in_feature_dim, in_feature_dim, edge_dim=edge_dim, add_self_loops=False, heads=heads, concat=False)
        self.conv2 = pyg_nn.GATv2Conv(in_feature_dim, in_feature_dim, edge_dim=edge_dim, add_self_loops=False, heads=heads, concat=False)

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
            2) Construct the feature vector consisting of {(inner, middle, and outer radius), (embedding), (learned_features)}
            2) Apply non-linearity to the volume and composition
            3) Apply non-linearity to the distances
        """
        
        edge_index, edge_attr = data.edge_index, data.edge_attr
        
        # Embedding the atom identity
        atom_identity = data.x[:, 0].long()
        x = self.embedder(atom_identity)
        additional_info = data.x[:, 1:]

        ## Add composition and volume information to the embedding
        # Step 1: MLP composition or not 
        if self.nonlinear_node_info:
            additional_info = self.node_mlp(additional_info)
            
        # Step 2: Concatenate or Multiply
        if self.concatenate_embedding:
            x = torch.concat([x, additional_info], dim=1)
        else:
            for i in range(0, 5):
                x = x * additional_info[:, i].expand(self.in_feature_dim, -1).transpose(0, 1)
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
                output = scatter(x, data.batch.unsqueeze(1), dim=0, reduce='sum')
            else:
                output = torch.sum(x, dim=0)
        return output
    
    def _evaluate_step(self, 
                       data):
        y_hat = self(data)
        try:
            y = data.y.view(data.num_graphs, -1)
        except:
            y = data.y
        loss = self.loss_function(y_hat, y)
        return y_hat, loss