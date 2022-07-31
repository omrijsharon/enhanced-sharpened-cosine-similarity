# https://mobile.twitter.com/_brohrer_/status/1232063619657093120?ref_src=twsrc%5Etfw%7Ctwcamp%5Etweetembed%7Ctwterm%5E1232063619657093120%7Ctwgr%5E%7Ctwcon%5Es1_&ref_url=https%3A%2F%2Frpisoni.dev%2Fposts%2Fcossim-convolution%2F
# https://github.com/brohrer/sharpened-cosine-similarity/blob/main/pytorch/sharpened_cosine_similarity.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import Module
from torch._torch_docs import reproducibility_notes

from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union


class SharpCosSim2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
    ) -> None:
        super(SharpCosSim2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype
        )
        # assert q_init > 0.0, "q_init must be positive"
        self.eps = torch.finfo(torch.float32).eps
        # random init p between 1 and 3 (starts from 1 with scale 2)
        p_scale = (3 - 1)
        p = (1 + p_scale * torch.rand(1, self.weight.size(1), 1, 1))
        self.log_p = nn.Parameter(torch.log(p))
        q_init = 1e-4
        self.log_q = torch.nn.Parameter(torch.full((1, 1, 1, 1), float(torch.log(torch.tensor(q_init)))))
        self.ones_kernel = torch.ones(self.kernel_size, device=self.weight.device, dtype=self.weight.dtype, requires_grad=False)

    def forward(self, inp: Tensor) -> Tensor:
        self.norm_weight_data()
        conv2d_out = F.conv2d(inp, self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        p = torch.exp(self.log_p)
        q = torch.exp(self.log_q)
        numerator = conv2d_out.abs()
        denominator = self.input_norm(inp) + q
        scs = torch.sign(conv2d_out) * (numerator / denominator) ** p
        return scs

    def norm_weight_data(self):
        self.weight.data /= torch.linalg.norm(self.weight.data, dim=(1, 2, 3), keepdim=True)

    def input_norm(self, inp):
        return torch.sqrt(F.conv2d(inp**2, self.ones_kernel, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) + self.eps)


if __name__ == '__main__':
    scs = SharpCosSim2d(6, 27, (3,3), groups=3)
    #@TODO: Test with groups!
    print()