
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch
from math import log10
import pytorch_ssim
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torch.autograd import Variable

def base_function(x, a=-0.5):
    # describe the base function sin(x)/x
    Wx = 0
    if np.abs(x) <= 1:
        Wx = (a+2)*(np.abs(x)**3) - (a+3)*x**2 + 1
    elif 1 <= np.abs(x) <= 2:
        Wx = a*(np.abs(x)**3) - 5*a*(np.abs(x)**2) + 8*a*np.abs(x) - 4*a
    return Wx


def padding(img):
    h, w, c = img.shape
    print(img.shape)
    pad_image = np.zeros((h+4, w+4, c))
    pad_image[2:h+2, 2:w+2] = img
    return pad_image


def draw_function():
    a = -0.5
    x = np.linspace(-3.0, 3.0, 100)
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = base_function(x[i], a)
    plt.figure("base_function")
    plt.plot(x, y)
    plt.show()


def bicubic(img, sacle, a=-0.5):
    print("Doing bicubic")
    h, w, color = img.shape
    img = padding(img)
    nh = h*sacle
    nw = h*sacle
    new_img = np.zeros((nh, nw, color))

    for c in range(color):
        for i in range(nw):
            for j in range(nh):

                px = i/sacle + 2
                py = j/sacle + 2
                px_int = int(px)
                py_int = int(py)
                u = px - px_int
                v = py - py_int

                A = np.matrix([[base_function(u+1, a)], [base_function(u, a)], [base_function(u-1, a)], [base_function(u-2, a)]])
                C = np.matrix([base_function(v+1, a), base_function(v, a), base_function(v-1, a), base_function(v-2, a)])
                B = np.matrix([[img[py_int-1, px_int-1][c], img[py_int-1, px_int][c], img[py_int-1, px_int+1][c], img[py_int-1, px_int+2][c]],
                               [img[py_int, px_int-1][c], img[py_int, px_int][c], img[py_int, px_int+1][c], img[py_int, px_int+2][c]],
                               [img[py_int+1, px_int-1][c], img[py_int+1, px_int][c], img[py_int+1, px_int+1][c], img[py_int+1, px_int+2][c]],
                               [img[py_int+2, px_int-1][c], img[py_int+2, px_int][c], img[py_int+2, px_int+1][c], img[py_int+2, px_int+2][c]]])
                new_img[j, i][c] = np.dot(np.dot(C, B), A)
    return new_img


def calc_psnr(img1, img2):
    return 10. * log10(img2.max()**2 / torch.mean((img1/1. - img2/1.) ** 2))


if __name__ == '__main__':

    sacle = 4
    path = "./data/1.png"

    hr_img = Image.open(path)
    w, h = hr_img.size
    lr_scale = Resize(min(w, h) // 4, interpolation=Image.BICUBIC)
    lr_img = lr_scale(hr_img)

    lr_img = cv2.cvtColor(np.asarray(lr_img), cv2.COLOR_RGB2BGR)

    sr_img = bicubic(lr_img, sacle)
    cv2.imwrite("img_001_bicubic.png", sr_img)

    sr_img = torch.from_numpy(sr_img.transpose([2, 0, 1])).unsqueeze(0)
    hr_img = torch.from_numpy(np.array(hr_img).transpose([2, 0, 1]).astype(np.double)).unsqueeze(0)
    print(sr_img.type())
    print(hr_img.type())
    # hr_img = Variable(ToTensor()(hr_img), volatile=True).unsqueeze(0)
    # hr_img = torch.DoubleTensor(hr_img)
    psnr = calc_psnr(sr_img, hr_img)
    ssim = pytorch_ssim.ssim(sr_img, hr_img)
    print('PSNR ：{:.2f}'.format(psnr))
    print('SSIM : {:.2f}'.format(ssim.item()))

    print("Finish")

