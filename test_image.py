import argparse
import time
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage, Resize

import pytorch_ssim
from model import Generator
from math import log10


if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='Test Single Image')
    # parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
    # parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
    # parser.add_argument('--image_name', type=str, help='test low resolution image name')
    # parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
    # opt = parser.parse_args()

    # UPSCALE_FACTOR = opt.upscale_factor
    # TEST_MODE = True if opt.test_mode == 'GPU' else False
    # IMAGE_NAME = opt.image_name
    # MODEL_NAME = opt.model_name

    def calc_psnr(img1, img2):
        return 10. * log10(img2.max()**2 / torch.mean((img1/1. - img2/1.) ** 2))

    UPSCALE_FACTOR = 4
    TEST_MODE = False
    IMAGE_NAME = 'data_13985.png'
    MODEL_NAME = 'tsrgan_netG_epoch_4_30.pth'

    model = Generator(UPSCALE_FACTOR).eval()
    if TEST_MODE:
        model.cuda()
        model.load_state_dict(torch.load('epochs/' + MODEL_NAME), strict=False)
    else:
        model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage), strict=False)

    hr_image = Image.open(IMAGE_NAME)
    w, h = hr_image.size

    with torch.no_grad():
        hr_image = Variable(ToTensor()(hr_image)).unsqueeze(0)

    lr_scale = Resize(min(w, h) // UPSCALE_FACTOR, interpolation=Image.BICUBIC)

    lr_image = lr_scale(hr_image)

    if TEST_MODE:
        hr_image = hr_image.cuda()

    start = time.perf_counter()
    sr_image = model(lr_image)
    elapsed = (time.perf_counter() - start)
    print('cost' + str(elapsed) + 's')

    sr_psnr = calc_psnr(hr_image, sr_image)
    print('sr_PSNR: {:.2f}'.format(sr_psnr))

    sr_ssim = pytorch_ssim.ssim(sr_image, hr_image)
    print('sr_SSIM: {:.2f}'.format(sr_ssim))

    sr_image = ToPILImage()(sr_image[0].data.cpu())
    sr_image.save('tsrgan_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)

    print('Finish')
