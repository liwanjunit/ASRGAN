
import torch
from PIL import Image
import pytorch_ssim
import torchvision.utils as utils
from torchvision.transforms import ToTensor, ToPILImage, Resize
from math import log10
from torch.autograd import Variable
from data_utils import display_transform
import matplotlib.pyplot as plt
from model.model_tsrgan import Generator_TSRGAN
from model.model_srcnn import SRCNN
from model.model import Generator

if __name__ == '__main__':

    CROP_SIZE = 256
    UPSCALE_FACTOR = 4

    image_name = 'data_5345.png'
    print('image_name: ' + image_name)
    print('----------------------')

    PATH = 'C:/code/ASRGAN/ASRGAN-master/test_image/compared/'

    SRCNN_MODEL_NAME = 'C:/code/SRCNN_Pytorch_1.0-master/SRCNN_Pytorch_1.0-master/outputs/x4/epoch_191.pth'
    SRGAN_MODEL_NAME = 'C:/code/train_results/new_model/srgan_x4/G/srgan_netG_epoch_4_193.pth'
    TSRGAN_MODEL_NAME = 'C:/code/train_results/new_model/tsrgan_x4/G/tsrgan_netG_epoch_4_177.pth'

    hr_path = 'C:/code/ASRGAN/ASRGAN-master/data/test_x4/target/'
    bicubic_path = 'C:/code/ASRGAN/ASRGAN-master/data/test_x4/bicubic/'

    srcnn_model = SRCNN().eval()
    srgan_model = Generator(UPSCALE_FACTOR).eval()
    tsrgan_model = Generator_TSRGAN(UPSCALE_FACTOR).eval()

    if torch.cuda.is_available():
        srcnn_model = srcnn_model.cuda()
        srgan_model = srgan_model.cuda()
        tsrgan_model = tsrgan_model.cuda()
    srcnn_model.load_state_dict(torch.load(SRCNN_MODEL_NAME), False)
    srgan_model.load_state_dict(torch.load(SRGAN_MODEL_NAME), False)
    tsrgan_model.load_state_dict(torch.load(TSRGAN_MODEL_NAME), False)

    hr_image = Image.open(hr_path + image_name)

    lr_scale = Resize(CROP_SIZE // UPSCALE_FACTOR, interpolation=Image.BICUBIC)
    bicubic_scale = Resize(CROP_SIZE, interpolation=Image.BICUBIC)
    lr_image = lr_scale(hr_image)
    bicubic_image = bicubic_scale(lr_image)

    lr_image = Variable(ToTensor()(lr_image), volatile=True).unsqueeze(0)
    hr_image = Variable(ToTensor()(hr_image), volatile=True).unsqueeze(0)
    bicubic_image = Variable(ToTensor()(bicubic_image), volatile=True).unsqueeze(0)

    # srcnn_image = Variable(srcnn_image, volatile=True).unsqueeze(0)
    # srgan_image = Variable(srgan_image, volatile=True).unsqueeze(0)
    # tsrgan_image = Variable(tsrgan_image, volatile=True).unsqueeze(0)

    if torch.cuda.is_available():
        lr_image = lr_image.cuda()
        hr_image = hr_image.cuda()
        bicubic_image = bicubic_image.cuda()

    srcnn_image = srcnn_model(bicubic_image)
    srgan_image = srgan_model(lr_image)
    tsrgan_image = tsrgan_model(lr_image)

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

    result = Image.open(PATH + image_name.split('.')[0] + '_compared' + '.png')

    plt.figure()
    plt.imshow(result)
    plt.show()

    print(f'Finish')
