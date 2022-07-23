import torch
import torchvision
import torchvision.transforms as transforms
import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
class Pixelshuffle1d_cstm(nn.Module):
    """Custom layer for Pixelshuffle in 1D"""
    def __init__(self, upscale_factor):
        super(Pixelshuffle1d_cstm, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self,x):
        batch_size, channels_in, samples_in = x.shape[0], x.shape[1], x.shape[2]
        channels_out = channels_in // self.upscale_factor
        samples_out = samples_in * self.upscale_factor

        out = x.contiguous().view([batch_size, self.upscale_factor, channels_out, samples_in])
        out = out.permute(0,2,3,1).contiguous()
        out = out.view(batch_size, channels_out, samples_out)

        return out

class Encoder(nn.Module):
    def __init__(self, frame_len,learn_h0):
        super(Encoder, self).__init__()
        self.res = []
        self.filters = [64, 128, 256]
        self.kernel_sz = [65,33,17]
        self.kernel_lst = [9]
        self.mxpool = nn.MaxPool1d(2)
        self.stride = 1
        self.up_stride = 2
        self.dconv01 = nn.Conv1d(in_channels=1, out_channels=self.filters[0], kernel_size=self.kernel_sz[0], stride=self.stride)
        self.dconv02 = nn.Conv1d(in_channels=self.filters[0], out_channels=self.filters[1], kernel_size=self.kernel_sz[1], stride=self.stride)
        self.dconv03 = nn.Conv1d(in_channels=self.filters[1], out_channels=self.filters[2], kernel_size=self.kernel_sz[2], stride=self.stride)

        self.rnn_01 = nn.GRU(input_size=128, hidden_size=128, num_layers=4, batch_first=True)

        self.utconv01 = nn.ConvTranspose1d(in_channels=self.filters[2], out_channels=self.filters[1], kernel_size=self.kernel_sz[2], stride=self.up_stride)
        self.dropout01 = nn.Dropout(0.5)

        self.utconv02 = nn.ConvTranspose1d(in_channels=2*self.filters[1], out_channels=self.filters[0],kernel_size=self.kernel_sz[1], stride=self.up_stride)
        self.dropout02 = nn.Dropout(0.5)

        self.utconv03 = nn.ConvTranspose1d(in_channels=2*self.filters[0], out_channels=1,kernel_size=self.kernel_sz[0], stride=self.up_stride)
        self.dropout03 = nn.Dropout(0.5)

        self.utconv04 = nn.ConvTranspose1d(in_channels=2, out_channels=1,kernel_size=self.kernel_sz[0], stride=self.up_stride)
        self.dropout04 = nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.lkyrelu = nn.LeakyReLU(0.2)

        h0 = torch.zeros(4,128)

        if learn_h0:
            self.h0 = torch.nn.Parameter(h0)


    def forward(self, x, hidden=None ):

        """Downsampling"""

        pad_w = max((math.ceil(x.shape[-1]//self.stride) - 1)*self.stride + (self.kernel_sz[0] - 1)*1+1-x.shape[-1],0)
        out = F.pad(x, [pad_w // 2, pad_w - pad_w // 2])
        out = self.lkyrelu(self.mxpool(self.dconv01(out)))
        self.res.append(out)

        pad_w = max((math.ceil(out.shape[-1]//self.stride) - 1)*self.stride + (self.kernel_sz[1] - 1)*1+1-out.shape[-1],0)
        out = F.pad(out, [pad_w // 2, pad_w - pad_w // 2])
        out = self.lkyrelu(self.mxpool(self.dconv02(out)))
        self.res.append(out)

        pad_w = max((math.ceil(out.shape[-1]//self.stride) - 1)*self.stride + (self.kernel_sz[2] - 1)*1+1-out.shape[-1],0)
        out = F.pad(out, [pad_w // 2, pad_w - pad_w // 2])
        out = self.lkyrelu(self.mxpool(self.dconv03(out)))
        # self.res.append(out)

        """LSTM Layer"""

        (out, hidden) = self.rnn_01(out, hidden)

        """Upsampling"""
        pad_w = self.kernel_sz[2]//2-1
        out = out[:,:,pad_w//2:-(pad_w-pad_w//2)]
        out = self.relu(self.dropout01(self.utconv01(out)[:,:,:-1]))
        out = torch.cat((out, self.res[-1]), dim=1)

        pad_w = self.kernel_sz[1]//2-1
        out = out[:,:,pad_w//2:-(pad_w-pad_w//2)]
        out = self.relu(self.dropout02(self.utconv02(out)[:,:,:-1]))
        out = torch.cat((out, self.res[0]), dim=1)

        pad_w = self.kernel_sz[0]//2-1
        out = out[:,:,pad_w//2:-(pad_w-pad_w//2)]
        out = self.relu(self.dropout03(self.utconv03(out)[:,:,:-1]))
        out = torch.cat((out, x), dim=1)

        pad_w = self.kernel_sz[0]//2-1
        out = out[:,:,pad_w//2:-(pad_w-pad_w//2)]
        out = self.utconv04(out)[:,:,:-1]

        self.res.clear()

        return out, hidden




