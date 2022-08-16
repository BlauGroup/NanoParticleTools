from multiprocessing.sharedctypes import Value
import torch
from torch.utils import data
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Callable, Optional, Union, List
import numpy as np


class SpectrumModelBase(pl.LightningModule):
    def __init__(self,
                 n_input_nodes: int,
                 n_output_nodes: Optional[int]=20,
                 nn_layers: Optional[List[int]] = [128],
                 dopants: Union[list, dict] = ['Yb', 'Er', 'Nd'],
                 learning_rate: Optional[float]=1e-5,
                 lr_scheduler: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler]]=None,
                 loss_function: Optional[Callable[[List, List], float]] = F.mse_loss,
                 l2_regularization_weight: float = 0,
                 dropout_probability: float = 0, 
                 optimizer_type: str = 'SGD',
                 **kwargs):
        super().__init__()

        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.l2_regularization_weight = l2_regularization_weight
        self.dropout_probability = dropout_probability
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function


        self.n_layers = len(nn_layers)
        self.nn_layers = nn_layers
        self.n_input_nodes = n_input_nodes
        self.n_output_nodes = n_output_nodes
        
        if isinstance(dopants, list):
            self.dopant_map = {key:i for i, key in enumerate(dopants)}
        elif isinstance(dopants, dict):
            self.dopant_map = dopants
        else:
            raise ValueError('Expected dopants to be of type list or dict')
        
        self.dopants = dopants
        
        # Build the Feed Forward Neural Network Architecture
        current_n_nodes = self.n_input_nodes
        layers = []
        for i, n_nodes in enumerate(self.nn_layers):
            layers.extend(self._get_layer(current_n_nodes, n_nodes))
            current_n_nodes = n_nodes
        layers.append(nn.Linear(current_n_nodes, self.n_output_nodes))
        # Add a final softplus to constrain outputs to be always positive
        layers.append(nn.ReLU())
        self.nn = nn.Sequential(*layers)
        
        self.save_hyperparameters()

    def _get_layer(self, n_input_nodes, n_output_nodes):
        _layers = []
        _layers.append(nn.Linear(n_input_nodes, n_output_nodes))
        _layers.append(nn.ReLU())
        if self.dropout_probability > 0:
            _layers.append(nn.Dropout(self.dropout_probability))       
        return _layers

    def configure_optimizers(self) -> Union[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
        """
        """ 
        # Default to the adam optimizer               
        if self.optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_regularization_weight)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_regularization_weight)

        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
        return [optimizer], [lr_scheduler]

    def forward(self, types, volumes, compositions):
        pass

    def _evaluate_step(self, 
                       batch):
        pass

    def training_step(self, 
                      batch, 
                      batch_idx):
        _, loss = self._evaluate_step(batch)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, 
                        batch, 
                        batch_idx):
        _, loss = self._evaluate_step(batch)
        self.log('val_loss', loss)
        return loss
        
    def test_step(self, 
                  batch, 
                  batch_idx):
        _, loss = self._evaluate_step(batch)
        self.log('test_loss', loss)
        return loss
        
    def predict_step(self, 
                     batch, 
                     batch_idx):
        pred, _ = self._evaluate_step(batch)
        return pred

    def predict_step_with_grad(self,
                               batch,
                               batch_idx):
        x, y = batch
        x = torch.tensor(x, requires_grad=True)
        pred = self(x)
        return pred

class SpectrumModel(SpectrumModelBase):
    def __init__(self, 
                 **kwargs):
        super().__init__(**kwargs)
    
    def describe(self):
        """
        Output a name that captures the hyperparameters used
        """
        descriptors = []

        # Type of optimizer
        if self.optimizer_type.lower() == 'sgd':
            descriptors.append('sgd')
        else:
            descriptors.append('adam')
        
        # Number of nodes/layers
        nodes = []
        nodes.append(self.n_input_nodes)
        nodes.extend(self.n_hidden_nodes)
        nodes.append(self.n_output_nodes)
        descriptors.append('-'.join([str(i) for i in nodes]))

        descriptors.append(f'lr-{self.learning_rate}')
        descriptors.append(f'dropout-{self.dropout_probability:.2f}')
        descriptors.append(f'l2_reg-{self.l2_regularization_weight:.2E}')
        
        return '_'.join(descriptors)

    def forward(self, x):
        return self.nn(x)
    
    def configure_optimizers(self) -> Union[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
        """
        """ 
        # Default to the adam optimizer               
        if self.optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_regularization_weight)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_regularization_weight)

        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
        return [optimizer], [lr_scheduler]

    def _evaluate_step(self, 
                       batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        return y_hat, loss

class SpectrumAttentionModel(SpectrumModelBase):
    def __init__(self,
                 embedding_dimension: Optional[int] = 12,
                 n_heads: Optional[int] = 4,
                 embedding_dropout: float = 0,
                 sum_attention_output: Optional[bool] = False,
                 **kwargs):
        """
        :param nn_layers: 
        :param dopants: List of Dopants in the model or a map of str->int
        :param embedding_dimension: size of embedding vector
        :param n_heads: Number of heads to use in the Multiheaded Attention step
        """
        super().__init__(n_input_nodes=embedding_dimension, **kwargs)

        self.embedding_dropout = embedding_dropout
        self.sum_attention_output = sum_attention_output
        
        self.embedding_dimension = embedding_dimension
        self.n_heads = n_heads
        
        self.embedding = nn.Embedding(len(self.dopants), self.embedding_dimension)
        self.embedding_dropout = nn.Dropout(self.embedding_dropout)
        self.multihead_attn = nn.MultiheadAttention(self.embedding_dimension, self.n_heads, batch_first=True)

        self.save_hyperparameters()

    def forward(self, types, volumes, compositions):
        types = types.to(self.device)
        volumes = volumes.to(self.device)
        compositions = compositions.to(self.device)

        # Perform the look-up to create the embedding vectors
        embedding = self.embedding(types)
        
        embedding = self.embedding_dropout(embedding)
        # Multiply by both volume and compositions.
        # This will have the effect of zero-ing out the embedding vector
        # where the dopant does not exist. Additionally, it will
        # add information on the size of the layer and the quantity of the dopant present
        embedding = embedding * compositions.unsqueeze(-1) * volumes.unsqueeze(-1)

        attn_output, _ = self.multihead_attn(embedding, embedding, embedding, need_weights=False)

        # Apply a mask
        ## First, compute the mask
        mask = torch.where(embedding == 0, torch.zeros(attn_output.size(), device=self.device), torch.ones(attn_output.size(), device=self.device))
        mask.to(self.device)

        ## Now we apply the mask
        masked_attn_output = attn_output * mask
        
        if self.sum_attention_output:
            x = torch.mean(masked_attn_output, dim=-2)
            output = self.nn(x)
        else:
            x = self.nn(masked_attn_output)
            output = torch.sum(x, dim=-2)
        return output

    def _evaluate_step(self, 
                       batch):
        (types, volumes, compositions), y = batch
        y_hat = self(types, volumes, compositions)
        loss = self.loss_function(y_hat, y)
        return y_hat, loss

class TransformerSpectrumModel(SpectrumModelBase):
    def __init__(self,
                 embedding_dimension: Optional[int] = 12,
                 n_heads: Optional[int] = 4,
                 n_encoders: Optional[int] = 2,
                 embedding_dropout: Optional[float] = 0,
                 transformer_dropout: Optional[float] = 0,
                 sum_attention_output: Optional[bool] = False,
                 **kwargs):
        if 'n_input_nodes' in kwargs:
            del kwargs['n_input_nodes']
        super().__init__(n_input_nodes=embedding_dimension, **kwargs)
        
        self.embedding_dimension = embedding_dimension
        self.n_heads = n_heads
        self.n_encoders = n_encoders
        self.transformer_dropout = transformer_dropout
        self.embedding_dropout = embedding_dropout
        self.sum_attention_output = sum_attention_output
        
        self.embedding = nn.Embedding(len(self.dopants), self.embedding_dimension)
        self.embedding_dropout = nn.Dropout(self.embedding_dropout)
        
        single_encoder_layer = nn.TransformerEncoderLayer(embedding_dimension, n_heads, batch_first=True, dropout=transformer_dropout)
        self.encoder = nn.TransformerEncoder(single_encoder_layer, n_encoders)
        
    def _evaluate_step(self, 
                       batch):
        (types, volumes, compositions), y = batch
        y_hat = self(types, volumes, compositions)
        loss = self.loss_function(y_hat, y)
        return y_hat, loss
    
    def forward(self, types, volumes, compositions):
        types = types.to(self.device)
        volumes = volumes.to(self.device)
        compositions = compositions.to(self.device)

        # Perform the look-up to create the embedding vectors
        embedding = self.embedding(types)
        
        embedding = self.embedding_dropout(embedding)
        # Multiply by both volume and compositions.
        # This will have the effect of zero-ing out the embedding vector
        # where the dopant does not exist. Additionally, it will
        # add information on the size of the layer and the quantity of the dopant present
        embedding = embedding * compositions.unsqueeze(-1) * volumes.unsqueeze(-1)
        
        # Use the TransformerEncoder to apply the attention mechanism
        attn_output = self.encoder(embedding)
        
        # Apply a mask
        ## First, compute the mask
        mask = torch.where(embedding == 0, torch.zeros(attn_output.size(), device=self.device), torch.ones(attn_output.size(), device=self.device))
        mask.to(self.device)

        ## Now we apply the mask
        masked_attn_output = attn_output * mask
        
        if self.sum_attention_output:
            x = torch.mean(masked_attn_output, dim=-2)
            output = self.nn(x)
        else:
            x = self.nn(masked_attn_output)
            output = torch.sum(x, dim=-2)
        return output