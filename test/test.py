
import torch
from PIL import Image
import pytorch_ssim
import torchvision.utils as utils
from torchvision.transforms import ToTensor, ToPILImage
from math import log10
from torch.autograd import Variable
from data_utils import display_transform
import matplotlib.pyplot as plt

if __name__ == '__main__':

    image_name = 'data_13985.png'
    print('image_name: ' + image_name)
    print('----------------------')

    PATH = 'C:/code/ASRGAN/ASRGAN-master/test_image/compared/'

    hr_image = ToTensor()(Image.open(PATH + 'hr/' + image_name))
    bicubic_image = ToTensor()(Image.open(PATH + 'bicubic/' + image_name))
    srcnn_image = ToTensor()(Image.open(PATH + 'srcnn/' + image_name))
    srgan_image = ToTensor()(Image.open(PATH + 'srgan/' + image_name))
    tsrgan_image = ToTensor()(Image.open(PATH + 'tsrgan/' + image_name))

    hr_image = Variable(hr_image, volatile=True).unsqueeze(0)
    bicubic_image = Variable(bicubic_image, volatile=True).unsqueeze(0)
    srcnn_image = Variable(srcnn_image, volatile=True).unsqueeze(0)
    srgan_image = Variable(srgan_image, volatile=True).unsqueeze(0)
    tsrgan_image = Variable(tsrgan_image, volatile=True).unsqueeze(0)

    bicubic_psnr = 10 * log10(1 / ((hr_image - bicubic_image) ** 2).data.mean())
    srcnn_psnr = 10 * log10(1 / ((hr_image - srcnn_image) ** 2).data.mean())
    srgan_psnr = 10 * log10(1 / ((hr_image - srgan_image) ** 2).data.mean())
    tsrgan_psnr = 10 * log10(1 / ((hr_image - tsrgan_image) ** 2).data.mean())
    print('bicubic_PSNR: {:.4f}'.format(bicubic_psnr))
    print('srcnn_PSNR:   {:.4f}'.format(srcnn_psnr))
    print('srgan_PSNR:   {:.4f}'.format(srgan_psnr))
    print('tsrgan_PSNR:  {:.4f}'.format(tsrgan_psnr))

    print('----------------------')

    bicubic_ssim = pytorch_ssim.ssim(bicubic_image, hr_image)
    srcnn_ssim = pytorch_ssim.ssim(srcnn_image, hr_image)
    srgan_ssim = pytorch_ssim.ssim(srgan_image, hr_image).item()
    tsrgan_ssim = pytorch_ssim.ssim(tsrgan_image, hr_image).item()
    print('bicubic_SSIM: {:.4f}'.format(bicubic_ssim))
    print('srcnn_SSIM:   {:.4f}'.format(srcnn_ssim))
    print('srgan_SSIM:   {:.4f}'.format(srgan_ssim))
    print('tsrgan_SSIM:  {:.4f}'.format(tsrgan_ssim))

    print('----------------------')

    test_images = torch.stack(
        [display_transform()(bicubic_image.data.cpu().squeeze(0)), display_transform()(srcnn_image.data.cpu().squeeze(0)),
         display_transform()(srgan_image.data.cpu().squeeze(0)), display_transform()(tsrgan_image.data.cpu().squeeze(0)),
         display_transform()(hr_image.data.cpu().squeeze(0))])
    image = utils.make_grid(test_images, nrow=5, padding=5)
    utils.save_image(image, PATH + image_name.split('.')[0] + '_compared' + '.png')



    print(f'Finish')
