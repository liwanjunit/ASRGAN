
import torch
from PIL import Image
import pytorch_ssim
import torchvision.utils as utils
from torchvision.transforms import ToTensor, ToPILImage
from math import log10
from torch.autograd import Variable
from data_utils import display_transform
import matplotlib.pyplot as plt
from model.model_srgan import Generator

if __name__ == '__main__':

    UPSCALE_FACTOR = 4

    HR_PATH = f'E:/code/dataset/Set5/set5_HR_{UPSCALE_FACTOR}/img_001_SRF_{UPSCALE_FACTOR}_HR.png'
    MATLAB_PATH = f'E:/code/dataset/Set5/set5_LR_{UPSCALE_FACTOR}/img_001_SRF_{UPSCALE_FACTOR}_LR.png'
    PYTHON_PATH = f'E:/code/dataset/Set5/set5_LR_{UPSCALE_FACTOR}_python/python_img_001_SRF_{UPSCALE_FACTOR}_HR.png'

    MODEL_NAME = f'E:/code/train_results/new_model/x{UPSCALE_FACTOR}/srgan_x{UPSCALE_FACTOR}/G/srgan_netG_epoch_{UPSCALE_FACTOR}_200.pth'

    model = Generator(UPSCALE_FACTOR).eval()
    if torch.cuda.is_available():
        model.cuda()
        model.load_state_dict(torch.load(MODEL_NAME), strict=False)
    else:
        model.load_state_dict(torch.load(MODEL_NAME, map_location=lambda storage, loc: storage), strict=False)

    hr_image = ToTensor()(Image.open(HR_PATH).convert('RGB'))
    matlab_image = ToTensor()(Image.open(MATLAB_PATH).convert('RGB'))
    python_image = ToTensor()(Image.open(PYTHON_PATH).convert('RGB'))

    hr_image = Variable(hr_image, volatile=True).unsqueeze(0)
    matlab_image = Variable(matlab_image, volatile=True).unsqueeze(0)
    python_image = Variable(python_image, volatile=True).unsqueeze(0)

    if torch.cuda.is_available():
        hr_image = hr_image.cuda()
        matlab_image = matlab_image.cuda()
        python_image = python_image.cuda()

    matlab_image = model(matlab_image)
    python_image = model(python_image)

    matlab_psnr = 10 * log10(1 / ((hr_image - matlab_image) ** 2).data.mean())
    python_psnr = 10 * log10(1 / ((hr_image - python_image) ** 2).data.mean())

    print('matlab_psnr:   {:.4f}'.format(matlab_psnr))
    print('python_psnr:   {:.4f}'.format(python_psnr))

    print('----------------------')

    matlab_ssim = pytorch_ssim.ssim(matlab_image, hr_image).item()
    python_ssim = pytorch_ssim.ssim(python_image, hr_image).item()

    print('matlab_ssim:   ', matlab_ssim)
    print('python_ssim:   ', python_ssim)

    print('----------------------')

    matlab_image = ToPILImage()(matlab_image[0].data.cpu())
    python_image = ToPILImage()(python_image[0].data.cpu())

    matlab_image.save(f'E:/code/dataset/Set5/MATLAB_SR.png')
    python_image.save(f'E:/code/dataset/Set5/PYTHON_SR.png')

    print(f'Finish')

