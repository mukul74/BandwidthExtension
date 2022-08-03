"""
SampleLevelMLP module for SampleRnn module
Date :  28 Dec 2021
Written By :  Mukul Agarwal 62380
"""
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SampleLevelMlp(nn.Module):
    def __init__(self,frame_size,dim,q_levels,embed_size):
        super(SampleLevelMlp, self).__init__()
        self.frame_size = frame_size
        self.dim = dim
        self.q_levels = q_levels
        self.embed_size = embed_size

        """Layers"""
        self.embedding_layer = nn.Embedding(num_embeddings=self.embed_size,embedding_dim=self.embed_size)
        self.inputs_layer = nn.Conv1d(in_channels=self.q_levels,out_channels=2*self.dim,kernel_size=self.frame_size, bias=True)
        self.hidden_layer_01 = nn.Linear(in_features=self.dim,out_features=self.dim)
        self.hidden_layer_02 = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.output_layer = nn.Linear(in_features=self.dim, out_features=self.q_levels)

    def forward(self,inputs, conditioning_frames):
        batch_size = inputs.shape[0]

        inputs = self.embedding_layer(inputs.contiguous().view(-1))
        inputs = self.inputs_layer(torch.reshape(inputs,(batch_size,self.q_levels,-1)))
        hidden = F.relu(self.hidden_layer_01(inputs + conditioning_frames))
        hidden = F.relu(self.hidden_layer_02(hidden))
        output = self.output_layer(hidden)
        return output