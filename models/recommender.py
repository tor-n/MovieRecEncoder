import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

class Recommender(nn.Module):
    def __init__(self, n_classes=1000, embedding_dim=127, projection_dim=256, user_info_len=30, movie_info_len=19, seq_len=len_train_sequence):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.movie_projection_dim = projection_dim
        self.user_projection_dim = embedding_dim+projection_dim+1

        self.user_projection = nn.Linear(user_info_len, self.user_projection_dim)
        self.movie_projection = nn.Linear(movie_info_len, self.movie_projection_dim)
        self.movie_embedding = nn.Embedding(n_classes, self.embedding_dim)

        self.transformer = Transformer(self.user_projection_dim)
        self.output_head = nn.Linear(self.user_projection_dim, n_classes)
        self.position_encoding = PositionEmbedding(self.user_projection_dim)        

    def forward(self, input):
        # input: sequences is a dictionary with three keys:
        # input key 1) "movie_seq": list of watch history of users. They are sequences of movie ids
        # input key 2) "movie_info": information about movies in the batch 
        # input key 3) "user_info": information about users in the batch

        user_info, movie_info, movie_seq, movie_rating = input
        if len(movie_rating.shape) == 2:
          movie_rating = torch.unsqueeze(movie_rating, 2)

        user_proj = self.user_projection(user_info) # size: (batch, user_projection_dim)
        user_proj = torch.unsqueeze(user_proj, 1) # size: (batch, 1, user_projection_dim)
        movie_proj = self.movie_projection(movie_info) # size: (batch, seq_len, movie_projection_dim)
        movie_emb = self.movie_embedding(movie_seq) # size: (batch, seq_len, embedding_dim)
        sequences = torch.cat([movie_emb, movie_proj, movie_rating], axis=2) # size: (batch, seq_len, user_projection_dim)
        sequences = self.position_encoding(sequences)
        sequences = torch.cat([user_proj, sequences], axis=1) # size: (batch, seq_len+1, user_projection_dim)
        out_transformer = self.transformer(sequences) # size: (batch, seq_len+1, user_projection_dim)
        out_mean = torch.mean(out_transformer, dim=1) # size: (batch, user_projection_dim)
        logits = self.output_head(out_mean) # size: (batch, n_classes)
        
        return logits 
