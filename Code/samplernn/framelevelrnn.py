"""
FrameLevelRNN module for SampleRnn module
Date :  28 Dec 2021
Written By :  Mukul Agarwal 62380
"""

import torch
import torchvision
import torchvision.transforms as transforms
import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FrameLevelRnn(nn.Module):
    def __init__(self, rnn_type, frame_size, num_low_tier_frames, num_layers,
                 dim, q_levels, skip_connection, dropout, learn_h0):
        super(FrameLevelRnn, self).__init__()
        self.rnn_type = rnn_type
        self.frame_size = frame_size
        self.num_lower_tier_frames = num_low_tier_frames
        self.num_layers = num_layers
        self.dim = dim
        self.q_levels = q_levels
        self.skip_connection = skip_connection
        self.dropout = dropout
        h0 = torch.zeros(num_layers,dim)

        if learn_h0:
            self.h0 = torch.nn.Parameter(h0)
        else:
            with torch.no_grad:
                self.register_buffer('h0',h0)

        """Layers"""
        self.input_layer = nn.Conv1d(in_channels=self.frame_size, out_channels=self.dim, kernel_size=1, stride=1, bias=True)
        # self.input_layer = nn.Linear(in_features=self.frame_size, out_features=self.dim)

        self.rnn_layer = nn.GRU(input_size=self.dim, hidden_size=self.dim, num_layers=self.num_layers,batch_first=True)
        self.upsample_layer = nn.ConvTranspose1d(in_channels=self.dim,out_channels=self.dim,kernel_size=self.num_lower_tier_frames,stride=self.num_lower_tier_frames, bias=True)


        """Basically FramelevelRnn will contain the input layer(Dense) -> RNN layer(GRU) -> Umpsampling layer(ConvTranspose1d)"""

    def forward(self,inputs, upper_tier_conditioning=None, hidden=None):
        # print("Inside FrameLevelRnn module")

        batch_size = inputs.shape[0]

        # Reshape the tensor to get the inputs as per the algo
        input_frames = torch.reshape(inputs,
            (batch_size,inputs.shape[2]//self.frame_size,
            self.frame_size)).contiguous()

        # Dequantize the input samples
        # input_frames_1 = utils.mu_law_dequantization(inputs,self.q_levels)
        input_frames = ((input_frames / (self.q_levels/2.0))-1.0)*2.0
        num_steps = input_frames.shape[1]

        # Expand layer for now it is dense layer, could be replaced by conv layer in the later modifications
        input_frames = input_frames.permute(0,2,1)
        input_frames = self.input_layer(input_frames)
        input_frames = input_frames.permute(0, 2, 1)
        # Checks for the upper tier conditioning if yes add them point wise
        if upper_tier_conditioning is not None:
            input_frames += upper_tier_conditioning

        if hidden is None:
            (n_rnn,_) = self.h0.size()
            hidden = self.h0.unsqueeze(1) \
                .expand(n_rnn, batch_size, self.dim) \
                .contiguous()

        (frames_outputs, hidden) = self.rnn_layer(input_frames,hidden)
        output_shape = [
            batch_size, num_steps*self.num_lower_tier_frames,self.dim]

        frames_outputs = self.upsample_layer(frames_outputs.permute(0,2,1))
        frames_outputs = frames_outputs.permute(0,2,1)

        return (frames_outputs, hidden)

