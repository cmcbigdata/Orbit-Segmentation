import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F

from model.TimeDistributedLayer import TimeDistributedConv2d, TimeDistributedMaxPool, TimeDistributedUpsampling
from model.BiConvLSTM import BiConvLSTM
from model.Attention import Attention_block
  
class DeepSequentialNet(nn.Module):
    def __init__(self, image_size, device):
        super(DeepSequentialNet, self).__init__()
        self.device = device

        self.encoding_block1 = nn.Sequential(
            TimeDistributedConv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False, dropout=False),
            nn.ELU(inplace=True),
            TimeDistributedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False, dropout=False),
            nn.ELU(inplace=True)
        )
        self.encoding_block2 = nn.Sequential(
            TimeDistributedConv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False, dropout=False),
            nn.ELU(inplace=True),
            TimeDistributedConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False, dropout=False),
            nn.ELU(inplace=True)
        )
        self.encoding_block3 = nn.Sequential(
            TimeDistributedConv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False, dropout=False),
            nn.ELU(inplace=True),
            TimeDistributedConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False, dropout=False),
            nn.ELU(inplace=True)
        )

        self.biCLSTM1 = BiConvLSTM(input_size=(image_size//(2**3),image_size//(2**3)), input_dim=256, hidden_dim=512, kernel_size=(3,3), num_layers=3, device=self.device)

        self.Attention3 = Attention_block(F_g=512, F_l=256, F_int=256)

        self.decoding_block3 = nn.Sequential(
            TimeDistributedConv2d(512+256, 256, kernel_size=3, stride=1, padding=1, bias=False, dropout=False),
            nn.ELU(inplace=True),
            TimeDistributedConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False, dropout=False),
            nn.ELU(inplace=True)
        )

        self.Attention2 = Attention_block(F_g=256, F_l=128, F_int=128)

        self.decoding_block2 = nn.Sequential(
            TimeDistributedConv2d(256+128, 128, kernel_size=3, stride=1, padding=1, bias=False, dropout=False),
            nn.ELU(inplace=True),
            TimeDistributedConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False, dropout=False),
            nn.ELU(inplace=True)
        )

        self.Attention1 = Attention_block(F_g=128, F_l=64, F_int=64)

        self.decoding_block1 = nn.Sequential(
            TimeDistributedConv2d(128+64, 64, kernel_size=3, stride=1, padding=1, bias=False, dropout=False),
            nn.ELU(inplace=True)
        )

        self.biCLSTM2 = BiConvLSTM(input_size=(image_size,image_size), input_dim=64, hidden_dim=64, kernel_size=(3,3), num_layers=3, device=self.device)
        
        self.maxpooling = TimeDistributedMaxPool(2, stride=2)
        self.upsampling = TimeDistributedUpsampling(scale_factor=2, mode='nearest')
        
        self.onebyoneConv = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, mode):
        ### encoding
        encoded_vol1 = self.encoding_block1(input)
        maxpooled_encoded_vol1 = self.maxpooling(encoded_vol1)
        encoded_vol2 = self.encoding_block2(maxpooled_encoded_vol1)
        maxpooled_encoded_vol2 = self.maxpooling(encoded_vol2)
        encoded_vol3 = self.encoding_block3(maxpooled_encoded_vol2)
        maxpooled_encoded_vol3 = self.maxpooling(encoded_vol3)

        lstm_vol1 = self.biCLSTM1(maxpooled_encoded_vol3)

        if mode == 'feature_extractor':    
            return lstm_vol1
        elif mode == 'segmentation':
            ### decoding
            up_vol3 = self.upsampling(lstm_vol1)
            attention3 = self.Attention3(up_vol3, encoded_vol3)
            concat_vol3 = torch.cat((attention3, up_vol3), 2)
            decoded_vol3 = self.decoding_block3(concat_vol3)

            up_vol2 = self.upsampling(decoded_vol3)
            attention2 = self.Attention2(up_vol2, encoded_vol2)
            concat_vol2 = torch.cat((attention2, up_vol2), 2)
            decoded_vol2 = self.decoding_block2(concat_vol2)

            up_vol1 = self.upsampling(decoded_vol2)
            attention1 = self.Attention1(up_vol1, encoded_vol1)
            concat_vol1 = torch.cat((attention1, up_vol1), 2)
            decoded_vol1 = self.decoding_block1(concat_vol1)

            ### LSTM
            lstm_vol2 = self.biCLSTM2(decoded_vol1)
            lstm_vol2 = torch.sum(lstm_vol2, 1)

            ### Last PANG!~
            synth_code = self.onebyoneConv(lstm_vol2)

            return synth_code

class Domain_Discriminator(nn.Module):

    def __init__(self, in_channels, num_domain):
        super(Domain_Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2, bias=False),
            nn.LeakyReLU(True), 

            nn.Conv2d(32, 64, 5, stride=2, padding=2, bias=False),
            nn.LeakyReLU(True), 

            nn.Conv2d(64, 128, 5, stride=2, padding=2, bias=False),
            nn.LeakyReLU(True)
        )

        self.dense = nn.Sequential(
            nn.Linear(128 * 1 * 1, 10),
            nn.ReLU(True),
            nn.Linear(10, num_domain)
        )

        

    def forward(self, x):
        x = torch.sum(x, 1)
        x = torch.squeeze(x, axis=1)

        x = self.conv(x)            # (batch, 128, 1, 1)

        x = x.view(x.size(0), -1)   # (batch, 128 * 1 * 1)
        x = self.dense(x)           # (batch, 2)
        return x