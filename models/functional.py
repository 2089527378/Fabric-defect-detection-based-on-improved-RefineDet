'''
Script provides functional interface for β mish activation function.
'''

# import pytorch
import torch
from torch import nn
import torch.nn.functional as F

def beta_mish(input, beta=1.5):
    '''
    Applies the β mish function element-wise:
    β mish(x) = x * tanh(ln((1 + exp(x))^β))

    See additional documentation for beta_mish class.
    '''
    return input * torch.tanh(torch.log(torch.pow((1+torch.exp(input)),beta)))
