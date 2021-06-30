import torch
import torch.nn as nn
from model.TimeDistributedLayer import TimeDistributedConv2d, TimeDistributedSigmoid

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = TimeDistributedConv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=False)
        self.W_x = TimeDistributedConv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.psi = TimeDistributedSigmoid(F_int, 1, kernel_size=1,stride=1,padding=0,bias=False)
        
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi