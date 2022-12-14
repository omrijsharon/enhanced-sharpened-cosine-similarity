import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import Module
from torch._torch_docs import reproducibility_notes

from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union
from sharpened_cosine_similarity import SharpCosSim2d
from hpf import HPF


class EnhancedSCS2d(nn.Module):
    def __init__(self, conv2d_layer: nn.Conv2d, scs2d_layer: SharpCosSim2d):
        super(EnhancedSCS2d, self).__init__()
        self.conv2d_layer = conv2d_layer
        self.scs2d_layer = scs2d_layer

    def forward(self, x):
        x_filtered = self.conv2d_layer(x)
        assert x_filtered.size() == x.size(), "Filtered and input tensors must have the same size, please change pad size of conv2d layer"
        x_diff = x_filtered - x
        x_scs = self.scs2d_layer(x_diff)
        return x_scs


class HPF_SCS2d(nn.Module):
    def __init__(self, hpf_layer: HPF, scs2d_layer: SharpCosSim2d):
        super(HPF_SCS2d, self).__init__()
        self.hpf_layer = hpf_layer
        self.scs2d_layer = scs2d_layer

    def forward(self, x):
        x_hpf = self.hpf(x)
        x_scs = self.scs2d_layer(x_hpf)
        return x_scs