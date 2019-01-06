import os
import matplotlib.pyplot as plt
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(100, d*2, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*2)
        self.deconv1_2 = nn.ConvTranspose2d(10, d*2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d*2)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    # forward method
    def forward(self, input, label):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))
        # x = F.relu(self.deconv4_bn(self.deconv4(x)))
        # x = F.tanh(self.deconv5(x))
        return x


# fixed noise & label
temp_z_ = torch.randn(10, 100)
fixed_z_ = temp_z_
fixed_y_ = torch.zeros(10, 1)
for i in range(9):
    fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
    temp = torch.ones(10, 1) + i
    fixed_y_ = torch.cat([fixed_y_, temp], 0)

fixed_z_ = fixed_z_.view(-1, 100, 1, 1)
fixed_y_label_ = torch.zeros(100, 10)
fixed_y_label_.scatter_(1, fixed_y_.type(torch.LongTensor), 1)
fixed_y_label_ = fixed_y_label_.view(-1, 10, 1, 1)
with torch.no_grad():
    if torch.cuda.is_available():
        fixed_z_, fixed_y_label_ = Variable(fixed_z_.cuda()), Variable(fixed_y_label_.cuda())
    else:
        fixed_z_, fixed_y_label_ = Variable(fixed_z_), Variable(fixed_y_label_)
def generate_num(num_str,show = False, save = False, path = 'result.png'):
    G.eval()
    test_images = G(fixed_z_, fixed_y_label_)  # shape [100,1,32,32]
    test_images = test_images.reshape(10, 10, 32, 32)
    print(len(num_str))
    for style_idx in range(10):
        for idx, x in enumerate(num_str):
            num = int(x)
            # 保存单字的图片
            tmp_img = test_images[num, style_idx].cpu().data.numpy()
            tmp_img = np.maximum(tmp_img, 0)
            tmp_img = tmp_img * 255
            tmp_img = np.around(tmp_img)
            if idx == 0:
                img_gray = tmp_img.astype(np.uint8)
            else:
                ###hstack()在行上合并
                img_gray = np.hstack((img_gray, tmp_img.astype(np.uint8)))
        ret, thresh = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if save:
            generate_num_path = root + 'Fixed_results/generate_num' + str(style_idx) + '.png'
            cv2.imwrite(generate_num_path, thresh)
        if show:
            cv2.imshow('2', thresh)
            cv2.waitKey()


def show_result(num_str, show = False, save = False, path = 'result.png'):
    # img = cv2.imread('lena.jpg')
    #
    # # 1.转成灰度图
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # cv2.imshow('img', img)
    # cv2.imshow('gray', img_gray)
    # cv2.waitKey(0)

    G.eval()
    test_images = G(fixed_z_, fixed_y_label_) # shape [100,1,32,32]
    test_images = test_images.reshape(10, 10, 32, 32)
    print(len(num_str))
    for idx,x in enumerate(num_str):
        num = int(x)
        # 保存单字的图片
        tmp_img = test_images[num, 0].cpu().data.numpy()
        tmp_img = np.maximum(tmp_img, 0)
        tmp_img = tmp_img * 255
        tmp_img = np.around(tmp_img)
        if idx == 0:
            img_gray = tmp_img.astype(np.uint8)
        else:
            ###hstack()在行上合并
            img_gray = np.hstack((img_gray, tmp_img.astype(np.uint8)))
    ret, thresh = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite('result.jpg', thresh)

    # 保存单字的图片
    tmp_img = test_images[0, 0].cpu().data.numpy()
    tmp_img= np.maximum(tmp_img, 0)
    tmp_img = tmp_img * 255
    tmp_img = np.around(tmp_img)
    img_gray = tmp_img.astype(np.uint8)

    cv2.imwrite('single-test.jpg', img_gray)
    # img = cv2.imread('single-test.jpg')
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使用Otsu自动阈值，注意用的是cv2.THRESH_BINARY_INV
    ret, thresh = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite('single-test111.jpg', thresh)
    cv2.imshow('2', thresh)
    cv2.waitKey()

    # size_figure_grid = 10
    # fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    # for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
    #     ax[i, j].get_xaxis().set_visible(False)
    #     ax[i, j].get_yaxis().set_visible(False)
    #
    # for k in range(10*10):
    #     i = k // 10
    #     j = k % 10
    #     ax[i, j].cla()
    #     ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(), cmap='gray')
    #     # plt.imshow(test_images[k, 0].cpu().data.numpy())  # Needs to be in row,col order
    #     # fixed_p = root + 'Fixed_results/test' + str(k) + '.png'
    #     # plt.savefig(fixed_p)
    #
    # label = 'Epoch {0}'.format(num_epoch)
    # fig.text(0.5, 0.04, label, ha='center')
    # plt.savefig(path)
    #
    # if show:
    #     plt.show()
    # else:
    #     plt.close()


# network
G = generator(128)
if torch.cuda.is_available():
    G.cuda()
# results save folder
root = 'MNIST_cDCGAN_results/'
model = 'MNIST_cDCGAN_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')
if torch.cuda.is_available():
    G.load_state_dict(torch.load(root + model + 'generator_param.pkl'))
else:
    G.load_state_dict(torch.load(root + model + 'generator_param.pkl', map_location='cpu'))

num_str = '15228842534'
fixed_p = root + 'Fixed_results/' + model + str(num_str) + '.png'
# show_result(num_str, save=True, path=fixed_p)
generate_num(num_str, save=True, path=fixed_p)

