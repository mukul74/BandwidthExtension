"""
SampleRnn module for SampleRnn Project
Date :  28 Dec 2021
Written By :  Mukul Agarwal 62380
"""

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from samplernn import framelevelrnn
from samplernn import samplelevelmlp



class SampleRNN(nn.Module):
    def __init__(self,batch_size, frame_size, q_levels, q_type, dim,
                 rnn_type, num_rnn_layers, seq_len, emb_size, skip_connection, rnn_dropout):

        super(SampleRNN,self).__init__()
        self.batch_size = batch_size
        self.frame_size = frame_size[0]
        self.big_frame_size = frame_size[1]

        self.q_levels = q_levels
        self.q_type = q_type
        self.dim = dim
        self.rnn_type = rnn_type
        self.num_rnn_layers = num_rnn_layers
        self.seq_len = seq_len
        self.emb_size = emb_size
        self.skip_connection = skip_connection
        self.rnn_dropout = rnn_dropout
        self.hidden_03 = None
        self.hidden_02 = None
        layers = []

        """Creating the FrameLevelRnn module Tier 3, Tier 3 is the 1st Module from Top to Down View"""
        self.framelevelrnn_03 = framelevelrnn.FrameLevelRnn(rnn_type=self.rnn_type,
                                                            frame_size=self.big_frame_size,
                                                            num_low_tier_frames=(self.big_frame_size//self.frame_size),
                                                            num_layers=self.num_rnn_layers,
                                                            dim = self.dim,
                                                            q_levels = self.q_levels,
                                                            skip_connection=self.skip_connection,
                                                            dropout=self.rnn_dropout,learn_h0=True)

        """Creating the FrameLevelRnn module Tier 2, Tier 2 is the 2nd Module from Top to Down View"""
        self.framelevelrnn_02 = framelevelrnn.FrameLevelRnn(rnn_type=self.rnn_type,
                                                            frame_size=self.frame_size,
                                                            num_low_tier_frames=self.frame_size*2,
                                                            num_layers=self.num_rnn_layers,
                                                            dim = self.dim,
                                                            q_levels=self.q_levels,
                                                            skip_connection=self.skip_connection,
                                                            dropout = self.rnn_dropout,learn_h0=True)

        """Creating the SampleLevelRnn module Tier 1, Tier 1 is the 3rd Module from Top to Down View"""
        self.samplelevelmlp_01 = samplelevelmlp.SampleLevelMlp(frame_size=self.frame_size,
                                                               dim=self.dim,
                                                               q_levels=self.q_levels,
                                                               embed_size=self.emb_size)




    def forward(self, inputs, training=True, temprature=1.0):
        """
        Forward implementation for the samplernn, layer wise implementation for the samplernn
        Code is self explanatory and more details are provide inside the each modules forward function
        """
        "Third tier or First tier from top view gets the first 1024 sample as input and the hidden data for rnn"
        if inputs[-1] == True:
            self.hidden_03 = None
            self.hidden_02 = None

        third_tier_output,hidden_03 = self.framelevelrnn_03(inputs[0][:,:,: -self.big_frame_size],hidden=self.hidden_03)
        self.hidden_03 = hidden_03.detach()



        second_tier_output,hidden_02 = self.framelevelrnn_02(inputs[0][:,:,self.big_frame_size-self.frame_size: -self.frame_size],
                                upper_tier_conditioning=third_tier_output,hidden = self.hidden_02)
        self.hidden_02 = hidden_02.detach()

        First_tier_output = self.samplelevelmlp_01(inputs[0][:,:,self.big_frame_size-self.frame_size:-1].type(torch.int64)
                                                   ,conditioning_frames=second_tier_output)

        final_output = First_tier_output.permute(0,2,1)
        return final_output
