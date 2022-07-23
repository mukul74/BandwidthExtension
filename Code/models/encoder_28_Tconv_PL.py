import torch
import torchvision
import torchvision.transforms as transforms
import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

class Encoder(nn.Module):
    def __init__(self, frame_len,learn_h0):
        super(Encoder, self).__init__()
        self.filters = [32,64,128]
        self.kernel_down = [7,5,3]
        self.stride = 1
        self.upstride = 2
        self.res = []
        self.mxpool = nn.MaxPool1d(2)
        self.conv01 = nn.Conv1d(in_channels=8, out_channels=self.filters[0], kernel_size=self.kernel_down[0], stride=self.stride)
        self.conv02 = nn.Conv1d(in_channels=self.filters[0], out_channels=self.filters[1], kernel_size=self.kernel_down[1], stride=self.stride)
        self.conv03 = nn.Conv1d(in_channels=self.filters[1], out_channels=self.filters[2], kernel_size=self.kernel_down[2], stride= self.stride)

        self.rnn_01 = nn.GRU(input_size=16, hidden_size=16, num_layers=4, batch_first=True)

        self.Tconv01 = nn.ConvTranspose1d(in_channels=self.filters[2], out_channels=self.filters[1], kernel_size=self.kernel_down[2], stride=self.upstride)
        self.Tconv02 = nn.ConvTranspose1d(in_channels=2*self.filters[1], out_channels=self.filters[0], kernel_size=self.kernel_down[1], stride=self.upstride)
        self.Tconv03 = nn.ConvTranspose1d(in_channels=2*self.filters[0], out_channels=8, kernel_size=self.kernel_down[0], stride=self.upstride)
        self.Tconv04 = nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=self.kernel_down[1], stride=self.upstride)

        self.dropout01 = nn.Dropout(0.5)
        self.dropout02 = nn.Dropout(0.5)
        self.dropout03 = nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.lkyrelu = nn.LeakyReLU(0.2)
        h0 = torch.zeros(4,16)

        if learn_h0:
            self.h0 = torch.nn.Parameter(h0)

    def forward(self, x,hidden=None):
        batch_size = x.shape[0]
        if hidden is None:
            (n_rnn, _) = self.h0.size()
            hidden = self.h0.unsqueeze(1).expand(4, batch_size, 16).contiguous()

        x_in = torch.reshape(x, (batch_size, 128, 8)).contiguous()
        x_in = x_in.permute(0, 2, 1)
        pad_w = max((math.ceil(x_in.shape[-1] // self.stride) - 1) * self.stride + (self.kernel_down[0] - 1) * 1 + 1 -
                    x_in.shape[-1], 0)
        x_out = F.pad(x_in, [pad_w // 2, pad_w - pad_w // 2])
        x_out = self.lkyrelu(self.mxpool(self.conv01(x_out)))
        self.res.append(x_out)

        pad_w = max((math.ceil(x_out.shape[-1] // self.stride) - 1) * self.stride + (self.kernel_down[1] - 1) * 1 + 1 -
                    x_out.shape[-1], 0)
        x_out = F.pad(x_out, [pad_w // 2, pad_w - pad_w // 2])
        x_out = self.lkyrelu(self.mxpool(self.conv02(x_out)))
        self.res.append(x_out)

        pad_w = max((math.ceil(x_out.shape[-1] // self.stride) - 1) * self.stride + (self.kernel_down[2] - 1) * 1 + 1 -
                    x_out.shape[-1], 0)
        x_out = F.pad(x_out, [pad_w // 2, pad_w - pad_w // 2])
        x_out = self.lkyrelu(self.mxpool(self.conv03(x_out)))

        (x_out, hidden) = self.rnn_01(x_out, hidden)

        pad_w = self.kernel_down[2]//2 - 1
        x_out = x_out[:,:,pad_w:]
        x_out = self.relu(self.dropout01(self.Tconv01(x_out)))
        x_out = x_out[:,:,1:]
        x_out = torch.cat((x_out, self.res[-1]), dim=1)

        pad_w = self.kernel_down[1] // 2 - 1
        x_out = x_out[:, :, pad_w :]
        x_out = self.relu(self.dropout02(self.Tconv02(x_out)))
        x_out = x_out[:, :, 1:]
        x_out = torch.cat((x_out, self.res[-2]), dim=1)

        pad_w = self.kernel_down[0] // 2 - 1
        x_out = x_out[:, :, pad_w :]
        x_out = self.relu(self.dropout03(self.Tconv03(x_out)))
        x_out = x_out[:, :, 1:]
        x_out = torch.cat((x_out, x_in), dim=1)

        pad_w = self.kernel_down[0] // 2 - 2
        x_out = x_out[:, :, pad_w :]
        x_out = self.Tconv04(x_out)
        x_out = x_out[:, :, 1:]
        x_out = x_out.permute(0,2,1)
        x_out = torch.flatten(x_out, start_dim=1, end_dim=-1)
        x_out = torch.unsqueeze(x_out, dim=1)
        self.res.clear()
        return x_out, hidden

