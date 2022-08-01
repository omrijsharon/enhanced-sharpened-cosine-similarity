import torch
from torch.nn import functional as F
from functools import partial


def pwscs_loss(y_hat, y, kernel_size, dilation=1, padding=0, stride=1, q=1e-6, p=1, patch_reduce_func=torch.sum, batch_reduce_func=torch.mean):
    """
    Compute the Patch-wise Sharpened Cosine Similarity loss between 2 tensors.
    Maps to cosine similarity between patches when q=0 and p=1.
    Args:
        y_hat: the predicted output.
        y: the ground truth.
        conv2d kwargs...
        q: the magnitude of the expected noise floor.
        p: sharpening parameter.
        patch_reduce_func: function to reduce the dimension of the patches. i.e. torch.sum, torch.mean, etc.
        batch_reduce_func: function to reduce the dimension of the batch. i.e. torch.sum, torch.mean, etc.
    Returns:
        The SCS loss.
    """
    assert p > 0, "p must be greater than 0."
    assert q > 0, "q must be greater than 0."
    patch_reduce_func = partial(patch_reduce_func, dim=2, keepdim=True)
    batch_reduce_func = partial(batch_reduce_func, dim=0, keepdim=True)
    eps = torch.finfo(torch.float32).eps
    y_hat_unf = F.unfold(y_hat, kernel_size, dilation=dilation, padding=padding, stride=stride)
    y_hat_unf = y_hat_unf / (y_hat_unf.norm(dim=1, keepdim=True) + q + eps)
    y_unf = F.unfold(y, kernel_size, dilation=dilation, padding=padding, stride=stride)
    y_unf = y_unf / (y_unf.norm(dim=1, keepdim=True) + q + eps)
    unf_pw_mul = y_hat_unf * y_unf
    scs = torch.sign(unf_pw_mul) * torch.abs((unf_pw_mul.sum(dim=1, keepdim=True)) ** p) # [batch, patch_size, n_patches]
    return batch_reduce_func(patch_reduce_func(scs)).squeeze()

