
import torch.nn as nn


from models import encoder_27_Tconv
from models import encoder_28_Tconv_PL

class BWExtender(nn.Module):
    def __init__(self,frame_size,overlap_len, kernel_size):
        super(BWExtender, self).__init__()
        self.frame_size = frame_size
        self.overlap_len = overlap_len
        self.hidden_01 = None
        self.model = encoder_27_Tconv.Encoder(frame_len=self.frame_size, learn_h0=True)
        # self.model = encoder_28_Tconv_PL.Encoder(frame_len=self.frame_size, learn_h0=True)


    def forward(self,x):

        if x[-1] == True:
            self.hidden_01 = None

        out,hidden_01 = self.model(x[0], hidden = self.hidden_01)
        self.hidden_01 = hidden_01.detach()

        return out