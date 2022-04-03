import argparse
import time
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage, Resize

import pytorch_ssim
from model import Generator
from math import log10


def calc_psnr(img1, img2):
    return 10. * log10(img2.max() ** 2 / torch.mean((img1 / 1. - img2 / 1.) ** 2))


if __name__ == '__main__':


    UPSCALE_FACTOR = 4
    TEST_MODE = False

    MODEL_NAME = 'tsrgan_netG_epoch_4_40.pth'

    model = Generator(UPSCALE_FACTOR).eval()
    if TEST_MODE:
        model.cuda()
        model.load_state_dict(torch.load('epochs/' + MODEL_NAME), strict=False)
    else:
        model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage), strict=False)

    for i in range(385):

        IMAGE_NAME = f'data_{i+13985}.png'
        HR_PATH = f'data/test/target/data_{i+13985}.png'
        LR_PATH = f'data/test/data/data_{i+13985}.png'
        SR_PATH = f'data/test/results/'

        hr_image = Image.open(HR_PATH)
        lr_image = Image.open(LR_PATH)

        hr_image = Variable(ToTensor()(hr_image), volatile=True)
        lr_image = Variable(ToTensor()(lr_image), volatile=True)

        # if torch.cuda.is_available():
        #     lr_image = lr_image.cuda()
        #     hr_image = hr_image.cuda()

        sr_image = model(lr_image)
        mse = ((hr_image - sr_image) ** 2).data.mean()
        psnr = 10 * log10(1 / mse)
        print(psnr)
        ssim = pytorch_ssim.ssim(sr_image, hr_image).item()
        print(ssim)

        sr_image = ToPILImage()(sr_image[0].data.cpu())
        sr_image.save(SR_PATH + 'tsrgan_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)

    # sr_psnr = calc_psnr(hr_image, sr_image)
    # print('sr_PSNR: {:.2f}'.format(sr_psnr))
    #
    # sr_ssim = pytorch_ssim.ssim(sr_image, hr_image)
    # print('sr_SSIM: {:.2f}'.format(sr_ssim))

    # sr_image = ToPILImage()(sr_image[0].data.cpu())
    # sr_image.save('tsrgan_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)
    #
    # lr_image = ToPILImage()(lr_image[0].data.cpu())
    # lr_image.save('tsrgan_lr_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)
    #
    # hr_image = ToPILImage()(hr_image[0].data.cpu())
    # hr_image.save('tsrgan_hr_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)




    print('Finish')


