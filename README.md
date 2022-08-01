# Enhanced Sharpened Cosine Similarity (SCS)
A pytorch implementation of Sharpened Cosine Similarity enhanced with filtering for better feature extraction. 

## In this repo
- New SCS module (a bit different from the original).
- SCS Loss function.
- Fully differentiable High-Pass Filter (HPF).
- Enhancement layers: chaining the difference between filter/conv2d output and the input following by an SCS layer.
- TBA: 1D and 3D SCS layers.

## What is Sharpened Cosine Similarity and where did it come from?
[Brandon Rohrer](https://github.com/brohrer) shared his [insight on Twitter](https://twitter.com/_brohrer_/status/1232063619657093120?lang=en) about convolution layers not being a good feature extractors, and explained his idea to improve it (using cool animation!). [Reference to the tweet](https://twitter.com/_brohrer_/status/1232063619657093120?lang=en).

## The original implementation and it's weakness
Heavily inspired by [Brandon Rohrer's](https://github.com/brohrer) github [repo of sharpened-cosine-similarity](https://github.com/brohrer/sharpened-cosine-similarity), I implemented my own SCS2d pytorch layer a bit differently. When using the layer on images, we run into an **SCS weakness**: In an average image- there are a lot of areas with the same color and no significant change (in other words: an average image is composed largely out of low frequencies):

![picture alt](https://docs.opencv.org/3.4/fft1.jpg "FFT2 magnitude")

Normalization by the same color significantly reduces the output signal from the SCS layer.

## Solution: The Enhancement
Inspired from [Canny Edge Detector](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html) where a gaussian blur is applied first and the edge detection filter later- I took two different approaches with one idea in common: First filter the input, then take the difference between the filtered input and the original.

1. Chain a convolution layer and an SCS layer:
    1. Filter the image with a conv2d layer.
    2. Substract the the original image from the filtered image (hopfully to lower the values of the low frequency areas).
    3. Apply the SCS layer on the output from 1.b.

2. Start with a (fully differentiable) High-Pass Filter (HPF) to enhance edges and reduce the values of the low frequency pixels, then pass the HPF output to the SCS layer.
    1. The HPF layer is made of a gaussian blue layer and the output is the difference between the blured image and the original.
    2. The HPF layer is given a kernel size as a hyperparameter and learns 'sigma' (the spread of the gaussian within its kernel).
    3. The HPF output will have small values in most of an average image.

## Patch-wise Sharpened Cosine Similarity Loss
Instead of using an L2 loss in auto-encoders and some GAN architectures, we can compare features in each patch of the images between the 2 images using the SCS approach:
* Divied the images into patches (each sliding window becomes a patch, see [pytorch unfold](https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html) for more info).
* Use the SCS formula between all pairs of patchs.
* Reduce patch dimensions with your desired function (sum, mean, max, etc...).
* Reduce batch dimensions with your desired function (sum, mean, max, etc...).


## How to use the modules?
### SharpCosSim2d
The use of the SharpCosSim2d module and pytorch Conv2d module is the completely the same.
No need to insert p and q hyperparameters- in this implementation they are learned parameters!
```
scs2d_layer = SharpCosSim2d(3, 27, kernel_size=(3,3))
```
### HPF
Heavily borrowed from torchvision.transforms.functional_tensor but in contrast of pytorch implementation- it is fully differentiable!
```
    img = torch.randn(8, 3, 224, 224)
    hpf_layer = HPF(kernel_size=3)
    output = hpf_layer(img)
```
### EnhancedSCS2d
```
    img = torch.randn(8, 3, 224, 224)
    conv2d_layer = nn.Conv2d(3, 27, kernel_size=(3,3))
    scs2d_layer = SharpCosSim2d(3, 27, kernel_size=(3,3))
    enhanced_scs = EnhancedSCS2d(conv2d_layer, scs2d_layer)
    output = enhanced_scs(img)
```
### HPF_SCS2d
```
    img = torch.randn(8, 3, 224, 224)
    hpf_layer = HPF(kernel_size=11)
    scs2d_layer = SharpCosSim2d(3, 27, kernel_size=(3,3))
    enhanced_scs = EnhancedSCS2d(hpf_layer, scs2d_layer)
    output = enhanced_scs(img)
```
### SCS Loss
Here we DO have to choose p and q. When p = 1 and q = 0, the Sharpened Cosine Similarity reduces to just Cosine Similarity.
```
pwscs_loss(y_hat, y, kernel_size, q=1e-6, p=1, patch_reduce_func=torch.sum, batch_reduce_func=torch.mean)
```

## requirements
Tested on python 3.8 and Pytorch 1.10.
