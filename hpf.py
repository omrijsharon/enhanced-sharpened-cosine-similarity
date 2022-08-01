import torch
from torch import nn
from torch import Tensor
from torch.nn.functional import conv2d, pad as torch_pad
from torchvision.transforms.functional_tensor import _assert_image_tensor, _cast_squeeze_in, _cast_squeeze_out

from typing import Optional, Tuple, List


def _get_gaussian_kernel1d(kernel_size: int, sigma: Tensor) -> Tensor:
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d


def _get_gaussian_kernel2d(
        kernel_size: List[int], sigma: Tensor, dtype: torch.dtype, device: torch.device
) -> Tensor:
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0]).to(device, dtype=dtype)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1]).to(device, dtype=dtype)
    kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d


def gaussian_blur(img: Tensor, kernel_size: int, sigma: Tensor) -> Tensor:
    if not (isinstance(img, torch.Tensor)):
        raise TypeError('img should be Tensor. Got {}'.format(type(img)))
    _assert_image_tensor(img)

    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    kernel_size = [kernel_size, kernel_size] # make kernel a square
    sigma = sigma.tile(2) # make sigma the same in both axes
    kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype, device=img.device)
    kernel = kernel.expand(img.shape[-3], 1, kernel.shape[0], kernel.shape[1])

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [kernel.dtype, ])

    # padding = (left, right, top, bottom)
    padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
    img = torch_pad(img, padding, mode="reflect")
    img = conv2d(img, kernel, groups=img.shape[-3])

    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img


class HPF(nn.Module):
    def __init__(self, kernel_size: int):
        super(HPF, self).__init__()
        self.log_sigma = nn.Parameter(torch.randn(1))
        self.kernel_size = kernel_size
        self.sigma = None

    def forward(self, img: Tensor) -> Tensor:
        self.sigma = torch.exp(self.log_sigma)
        return gaussian_blur(img, kernel_size=self.kernel_size, sigma=self.sigma) - img


if __name__ == '__main__':
    hpf = HPF(kernel_size=3)
    img = torch.randn(8, 3, 10, 10)
    print(hpf(img).size())
