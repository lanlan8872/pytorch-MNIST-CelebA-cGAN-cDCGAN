# pytorch-MNIST-CelebA-cGAN-cDCGAN
Pytorch implementation of conditional Generative Adversarial Networks (cGAN) [1] and conditional Generative Adversarial Networks (cDCGAN) for MNIST [2] and CelebA [3] datasets.

* The network architecture (number of layer, layer size and activation function etc.) of this code differs from the paper.

* CelebA dataset used gender lable as condition.

* If you want to train using cropped CelebA dataset, you have to change isCrop = False to isCrop = True.

* you can download
  - MNIST dataset: http://yann.lecun.com/exdb/mnist/
  - CelebA dataset: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

## Implementation details
* cGAN

![GAN](pytorch_cGAN.png)

* cDCGAN

![Loss](pytorch_cDCGAN.png)

## Resutls
### MNIST
* Generate using fixed noise (fixed_z_)

<table align='center'>
<tr align='center'>
<td> cGAN</td>
<td> cDCGAN</td>
</tr>
<tr>
<td><img src = 'MNIST_cGAN_results/generation_animation.gif'>
<td><img src = 'MNIST_cDCGAN_results/MNIST_cDCGAN_generation_animation.gif'>
</tr>
</table>

* MNIST vs Generated images

<table align='center'>
<tr align='center'>
<td> MNIST </td>
<td> cGAN after 50 epochs </td>
<td> cDCGAN after 20 epochs </td>
</tr>
<tr>
<td><img src = 'MNIST_cGAN_results/raw_MNIST.png'>
<td><img src = 'MNIST_cGAN_results/MNIST_cGAN_50.png'>
<td><img src = 'MNIST_cDCGAN_results/MNIST_cDCGAN_20.png'>
</tr>
</table>

* Learning Time
  * MNIST cGAN - Avg. per epoch: 9.13 sec; Total 50 epochs: 937.06 sec
  * MNIST cDCGAN - Avg. per epoch: 47.16 sec; Total 20 epochs: 1024.26 sec

### CelebA
* Generate using fixed noise (fixed_z_; odd line - female (y: 0) & even line - male (y: 1); each two lines have the same style (1-2) & (3-4).)

<table align='center'>
<tr align='center'>
<td> cDCGAN</td>
<td> cDCGAN crop</td>
</tr>
<tr>
<td><img src = 'CelebA_cDCGAN_results/CelebA_cDCGAN_generation_animation.gif'>
<td><img src = 'CelebA_cDCGAN_crop_results/CelebA_cDCGAN_crop_generation_animation.gif'>
</tr>
</table>

* CelebA vs Generated images

<table align='center'>
<tr align='center'>
<td> CelebA </td>
<td> cDCGAN after 20 epochs </td>
<td> cDCGAN crop after 30 epochs </td>
</tr>
<tr>
<td><img src = 'CelebA_cDCGAN_results/raw_CelebA.png'>
<td><img src = 'CelebA_cDCGAN_results/CelebA_cDCGAN_20.png'>
<td><img src = 'CelebA_cDCGAN_crop_results/CelebA_cDCGAN_crop_30.png'>
</tr>
</table>

* CelebA cDCGAN morphing (noise interpolation)
<table align='center'>
<tr align='center'>
<td> cDCGAN </td>
<td> cDCGAN crop </td>
</tr>
<tr>
<td><img src = 'CelebA_cDCGAN_results/CelebA_cDCGAN_morp.png'>
<td><img src = 'CelebA_cDCGAN_crop_results/CelebA_cDCGAN_crop_morp.png'>
</tr>
</table>

* Learning Time
  * CelebA cDCGAN - Avg. per epoch: 826.69 sec; total 20 epochs ptime: 16564.10 sec

##图像纠偏
最常见的文本纠偏算法有两种，分别是

1.基于FFT变换以后频率域梯度
2.基于离散点求最小外接轮廓
这两种方法各有千秋，相对来说，第二种方法得到的结果更加准确，第一种基于离散傅立叶变换求振幅的方法有时候各种阈值选择在实际项目中会有很大问题。

## Development Environment

* Ubuntu 16.04 LTS
* NVIDIA GTX 1080 ti
* cuda 9.0
* Python 3.6.2
* certifi (2016.2.28)
* cffi (1.10.0)
* cycler (0.10.0)
* imageio (2.4.1)
* kiwisolver (1.0.1)
* matplotlib (3.0.2)
* mkl-fft (1.0.6)
* mkl-random (1.0.1)
* numpy (1.15.4)
* olefile (0.44)
* opencv-python (3.4.5.20)
* Pillow (4.2.1)
* pip (9.0.1)
* pycparser (2.18)
* pyparsing (2.3.0)
* python-dateutil (2.7.5)
* setuptools (36.4.0)
* six (1.10.0)
* torch (1.0.0)
* torchvision (0.2.1)
* wheel (0.29.0)
* wincertstore (0.2)
## Reference

[1] Mirza, Mehdi, and Simon Osindero. "Conditional generative adversarial nets." arXiv preprint arXiv:1411.1784 (2014).

(Full paper: https://arxiv.org/pdf/1411.1784.pdf)

[2] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998.

[3] Liu, Ziwei, et al. "Deep learning face attributes in the wild." Proceedings of the IEEE International Conference on Computer Vision. 2015.
