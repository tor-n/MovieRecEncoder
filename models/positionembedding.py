import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

class PositionEmbedding(nn.Module):
    def __init__(self, hidden_size, pos_factor=10000):
        super(PositionEmbedding, self).__init__()

        assert hidden_size % 2 == 0 and 'Model vector size must be even for sinusoidal encoding'
        power = torch.arange(0, hidden_size, step=2, dtype=torch.float32)[:] / hidden_size
        divisor = pos_factor ** power
        self.divisor = divisor
        self.hidden_size = hidden_size

    def forward(self, inputs, start=1):
        """
            Args:
                inputs: a float32 Tensor with shape [batch_size, sequence_length, hidden_size]

            Returns:
                embedding: a float32 Tensor with shape [batch_size, sequence_length, hidden_size]
        """
        assert inputs.shape[-1] == self.hidden_size and 'Input final dim must match model hidden size'

        batch_size = inputs.shape[0]
        sequence_length = inputs.shape[1]

        seq_pos = torch.arange(start, sequence_length + start, dtype=torch.float32)
        seq_pos_expanded = seq_pos[None,:,None]
        index = seq_pos_expanded.repeat(*[1,1,self.hidden_size//2])
        
        sin_embedding = torch.sin(index/self.divisor)
        cos_embedding = torch.cos(index/self.divisor)

        position_shape = (1, sequence_length, self.hidden_size) # fill in the other two dimensions
        position_embedding = torch.stack((sin_embedding,cos_embedding), dim=3).view(position_shape)

        pos_embed_deviced = position_embedding.to(get_device())
        return  inputs + pos_embed_deviced # add the embedding to the input
