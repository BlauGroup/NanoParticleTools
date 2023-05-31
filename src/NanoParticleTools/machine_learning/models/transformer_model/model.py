import torch
from torch import nn
from typing import Optional
from NanoParticleTools.machine_learning.models.mlp_model.model import MLPSpectrumModel


class TransformerSpectrumModel(MLPSpectrumModel):

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

        self.embedding = nn.Embedding(len(self.dopants),
                                      self.embedding_dimension)
        self.embedding_dropout = nn.Dropout(self.embedding_dropout)

        single_encoder_layer = nn.TransformerEncoderLayer(
            embedding_dimension,
            n_heads,
            batch_first=True,
            dropout=transformer_dropout)
        self.encoder = nn.TransformerEncoder(single_encoder_layer, n_encoders)

    def forward(self, x, **kwargs):
        types, volumes, compositions = x[:, 0].long(), x[:, 1], x[:, 2]

        # Perform the look-up to create the embedding vectors
        embedding = self.embedding(types)

        embedding = self.embedding_dropout(embedding)
        # Multiply by both volume and compositions.
        # This will have the effect of zero-ing out the embedding vector
        # where the dopant does not exist. Additionally, it will
        # add information on the size of the layer and the quantity of the dopant present
        embedding = embedding * compositions.unsqueeze(-1) * volumes.unsqueeze(
            -1)

        # Use the TransformerEncoder to apply the attention mechanism
        attn_output = self.encoder(embedding)

        # First, compute the mask
        mask = torch.where(embedding == 0,
                           torch.zeros(attn_output.size(), device=self.device),
                           torch.ones(attn_output.size(), device=self.device))
        mask.to(self.device)

        # Now we apply the mask
        masked_attn_output = attn_output * mask

        if self.sum_attention_output:
            x = torch.mean(masked_attn_output, dim=-2)
            output = self.nn(x)
        else:
            x = self.nn(masked_attn_output)
            output = torch.sum(x, dim=-2)
        return output


class SpectrumAttentionModel(MLPSpectrumModel):

    def __init__(self,
                 embedding_dimension: Optional[int] = 12,
                 n_heads: Optional[int] = 4,
                 embedding_dropout: float = 0,
                 sum_attention_output: Optional[bool] = False,
                 **kwargs):
        """_summary_

        Args:
            embedding_dimension (Optional[int], optional): size of embedding vector.
                Defaults to 12.
            n_heads (Optional[int], optional): Number of heads to use in the
                Multiheaded Attention step. Defaults to 4.
            embedding_dropout (float, optional): _description_.
                Defaults to 0.
            sum_attention_output (Optional[bool], optional): Whether to sum the attention
                outputs or concatenate (if False). Defaults to False.
        """
        if 'n_input_nodes' in kwargs:
            del kwargs['n_input_nodes']

        super().__init__(n_input_nodes=embedding_dimension, **kwargs)

        self.embedding_dropout = embedding_dropout
        self.sum_attention_output = sum_attention_output

        self.embedding_dimension = embedding_dimension
        self.n_heads = n_heads

        self.embedding = nn.Embedding(len(self.dopants),
                                      self.embedding_dimension)
        self.embedding_dropout = nn.Dropout(self.embedding_dropout)
        self.multihead_attn = nn.MultiheadAttention(self.embedding_dimension,
                                                    self.n_heads,
                                                    batch_first=True)

        self.save_hyperparameters()

    def forward(self, data):
        types, volumes, compositions = data.x[:, 0].long(), data.x[:, 1], data.x[:, 2]

        # Perform the look-up to create the embedding vectors
        embedding = self.embedding(types)

        embedding = self.embedding_dropout(embedding)
        # Multiply by both volume and compositions.
        # This will have the effect of zero-ing out the embedding vector
        # where the dopant does not exist. Additionally, it will
        # add information on the size of the layer and the quantity of the dopant present
        embedding = embedding * compositions.unsqueeze(-1) * volumes.unsqueeze(
            -1)

        attn_output, _ = self.multihead_attn(embedding,
                                             embedding,
                                             embedding,
                                             need_weights=False)

        # First, compute the mask
        mask = torch.where(embedding == 0,
                           torch.zeros(attn_output.size(), device=self.device),
                           torch.ones(attn_output.size(), device=self.device))
        mask.to(self.device)

        # Now we apply the mask
        masked_attn_output = attn_output * mask

        if self.sum_attention_output:
            x = torch.mean(masked_attn_output, dim=-2)
            output = self.nn(x)
        else:
            x = self.nn(masked_attn_output)
            output = torch.sum(x, dim=-2)
        return output
