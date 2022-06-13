
import torch
import pandas as pd
import pytorch_ssim
from PIL import Image
from torchvision.transforms import ToTensor, Resize
from math import log10
from torch.autograd import Variable
from model.model_srcnn import SRCNN
from model.model_srgan import Generator
from model.model_asrgan import Generator_ASRGAN
from model.model_esdr import EDSR
from  model.model_sasrgan import Generator_SASRGAN


if __name__ == '__main__':

    CROP_SIZE = 512
    UPSCALE_FACTOR = 4

    xmin = 796
    ymin = 896
    xmax = xmin + CROP_SIZE
    ymax = ymin + CROP_SIZE
    box = (xmin, ymin, xmax, ymax)

    image_name = 'NGC4631.png'
    print('image_name: ' + image_name)
    print('----------------------')

    PATH = 'E:/code/ASRGAN/ASRGAN-master/test/compared/'

    # hr_path = PATH + 'test/'
    hr_path = PATH + 'new/'

    srcnn_model = SRCNN().eval()
    edsr_model = EDSR(UPSCALE_FACTOR).eval()
    srresnet_model = Generator(UPSCALE_FACTOR).eval()
    srgan_model = Generator(UPSCALE_FACTOR).eval()
    asrresnet_model = Generator_ASRGAN(UPSCALE_FACTOR).eval()
    asrgan_model = Generator_ASRGAN(UPSCALE_FACTOR).eval()
    sasrgan_model = Generator_SASRGAN(UPSCALE_FACTOR).eval()

    img = Image.open(hr_path + image_name).convert('RGB')
    hr_image = img.crop(box)

    lr_scale = Resize(CROP_SIZE // UPSCALE_FACTOR, interpolation=Image.BICUBIC)
    bicubic_scale = Resize(CROP_SIZE, interpolation=Image.BICUBIC)
    lr_image = lr_scale(hr_image)
    bicubic_image = bicubic_scale(lr_image)

    lr_image = Variable(ToTensor()(lr_image), volatile=True).unsqueeze(0)
    hr_image = Variable(ToTensor()(hr_image), volatile=True).unsqueeze(0)
    bicubic_image = Variable(ToTensor()(bicubic_image), volatile=True).unsqueeze(0)

    if torch.cuda.is_available():
        srcnn_model = srcnn_model.cuda()
        edsr_model = edsr_model.cuda()
        srresnet_model = srresnet_model.cuda()
        srgan_model = srgan_model.cuda()
        asrresnet_model = asrresnet_model.cuda()
        asrgan_model = asrgan_model.cuda()
        sasrgan_model = sasrgan_model.cuda()

        lr_image = lr_image.cuda()
        hr_image = hr_image.cuda()
        bicubic_image = bicubic_image.cuda()

    srcnn_best_psnr = 0
    edsr_best_psnr = 0
    srresnet_best_psnr = 0
    srgan_best_psnr = 0
    asrresnet_best_psnr = 0
    asrgan_best_psnr = 0
    sasrgan_best_psnr = 0

    srcnn_best_ssim = 0
    edsr_best_ssim = 0
    srresnet_best_ssim = 0
    srgan_best_ssim = 0
    asrresnet_best_ssim = 0
    asrgan_best_ssim = 0
    sasrgan_best_ssim = 0

    srcnn_best_psnr_model = 0
    edsr_best_psnr_model = 0
    srresnet_best_psnr_model = 0
    srgan_best_psnr_model = 0
    asrresnet_best_psnr_model = 0
    asrgan_best_psnr_model = 0
    sasrgan_best_psnr_model = 0

    srcnn_best_ssim_model = 0
    edsr_best_ssim_model = 0
    srresnet_best_ssim_model = 0
    srgan_best_ssim_model = 0
    asrresnet_best_ssim_model = 0
    asrgan_best_ssim_model = 0
    sasrgan_best_ssim_model = 0

    for i in range(100):

        ASRResNet_MODEL_NAME = f'E:/code/train_results/new_model/x{UPSCALE_FACTOR}/asrresnet_x{UPSCALE_FACTOR}/model/asrresnet_epoch_{UPSCALE_FACTOR}_{i+1}.pth'
        ASRGAN_MODEL_NAME = f'E:/code/train_results/new_model/x{UPSCALE_FACTOR}/asrgan_x{UPSCALE_FACTOR}/G/asrgan_netG_epoch_{UPSCALE_FACTOR}_{i+1}.pth'
        SASRGAN_MODEL_NAME = f'E:/code/train_results/new_model/x{UPSCALE_FACTOR}/sasrgan_x{UPSCALE_FACTOR}/G/sasrgan_netG_epoch_{UPSCALE_FACTOR}_{i+1}.pth'

        asrresnet_model.load_state_dict(torch.load(ASRResNet_MODEL_NAME), False)
        asrgan_model.load_state_dict(torch.load(ASRGAN_MODEL_NAME), False)
        sasrgan_model.load_state_dict(torch.load(SASRGAN_MODEL_NAME), False)

        with torch.no_grad():

            asrresnet_image = asrresnet_model(lr_image)
            asrgan_image = asrgan_model(lr_image)
            sasrgan_image = asrgan_model(lr_image)

            asrresnet_psnr = 10 * log10(1 / ((hr_image - asrresnet_image) ** 2).data.mean())
            asrgan_psnr = 10 * log10(1 / ((hr_image - asrgan_image) ** 2).data.mean())
            sasrgan_psnr = 10 * log10(1 / ((hr_image - asrgan_image) ** 2).data.mean())

            asrresnet_ssim = pytorch_ssim.ssim(asrresnet_image, hr_image).item()
            asrgan_ssim = pytorch_ssim.ssim(asrgan_image, hr_image).item()
            sasrgan_ssim = pytorch_ssim.ssim(asrgan_image, hr_image).item()

            if asrresnet_psnr > asrresnet_best_psnr:
                asrresnet_best_psnr = asrresnet_psnr
                asrresnet_best_psnr_model = i + 1
            if asrgan_psnr > asrgan_best_psnr:
                asrgan_best_psnr = asrgan_psnr
                asrgan_best_psnr_model = i + 1
            if sasrgan_psnr > sasrgan_best_psnr:
                sasrgan_best_psnr = sasrgan_psnr
                sasrgan_best_psnr_model = i + 1
            if asrresnet_ssim > asrresnet_best_ssim:
                asrresnet_best_ssim = asrresnet_ssim
                asrresnet_best_ssim_model = i + 1
            if asrgan_ssim > asrgan_best_ssim:
                asrgan_best_ssim = asrgan_ssim
                asrgan_best_ssim_model = i + 1
            if sasrgan_ssim > sasrgan_best_ssim:
                sasrgan_best_ssim = sasrgan_ssim
                sasrgan_best_ssim_model = i + 1

    for i in range(100):

        SRCNN_MODEL_NAME = f'E:/code/train_results/new_model/x{UPSCALE_FACTOR}/srcnn_x{UPSCALE_FACTOR}/model/srcnn_epoch_{UPSCALE_FACTOR}_{i+1}.pth'
        EDSR_MODEL_NAME = f'E:/code/train_results/new_model/x{UPSCALE_FACTOR}/edsr_x{UPSCALE_FACTOR}/model/edsr_epoch_{UPSCALE_FACTOR}_{i+1}.pth'
        SRResNet_MODEL_NAME = f'E:/code/train_results/new_model/x{UPSCALE_FACTOR}/srresnet_x{UPSCALE_FACTOR}/model/srresnet_epoch_{UPSCALE_FACTOR}_{i+1}.pth'
        SRGAN_MODEL_NAME = f'E:/code/train_results/new_model/x{UPSCALE_FACTOR}/srgan_x{UPSCALE_FACTOR}/G/srgan_netG_epoch_{UPSCALE_FACTOR}_{i+1}.pth'

        srcnn_model.load_state_dict(torch.load(SRCNN_MODEL_NAME), False)
        edsr_model.load_state_dict(torch.load(EDSR_MODEL_NAME), False)
        srresnet_model.load_state_dict(torch.load(SRResNet_MODEL_NAME), False)
        srgan_model.load_state_dict(torch.load(SRGAN_MODEL_NAME), False)

        with torch.no_grad():

            srcnn_image = srcnn_model(bicubic_image)
            edsr_image = edsr_model(lr_image)
            srresnet_image = srresnet_model(lr_image)
            srgan_image = srgan_model(lr_image)

            srcnn_psnr = 10 * log10(1 / ((hr_image - srcnn_image) ** 2).data.mean())
            edsr_psnr = 10 * log10(1 / ((hr_image - edsr_image) ** 2).data.mean())
            srresnet_psnr = 10 * log10(1 / ((hr_image - srresnet_image) ** 2).data.mean())
            srgan_psnr = 10 * log10(1 / ((hr_image - srgan_image) ** 2).data.mean())

            srcnn_ssim = pytorch_ssim.ssim(srcnn_image, hr_image).item()
            edsr_ssim = pytorch_ssim.ssim(edsr_image, hr_image).item()
            srresnet_ssim = pytorch_ssim.ssim(srresnet_image, hr_image).item()
            srgan_ssim = pytorch_ssim.ssim(srgan_image, hr_image).item()

            if srcnn_psnr > srcnn_best_psnr:
                srcnn_best_psnr = srcnn_psnr
                srcnn_best_psnr_model = i + 1
            if edsr_psnr > edsr_best_psnr:
                edsr_best_psnr = edsr_psnr
                edsr_best_psnr_model = i + 1
            if srresnet_psnr > srresnet_best_psnr:
                srresnet_best_psnr = srresnet_psnr
                srresnet_best_psnr_model = i + 1
            if srgan_psnr > srgan_best_psnr:
                srgan_best_psnr = srgan_psnr
                srgan_best_psnr_model = i + 1

            if srcnn_ssim > srcnn_best_ssim:
                srcnn_best_ssim = srcnn_ssim
                srcnn_best_ssim_model = i + 1
            if edsr_ssim > edsr_best_ssim:
                edsr_best_ssim = edsr_ssim
                edsr_best_ssim_model = i + 1
            if srresnet_ssim > srresnet_best_ssim:
                srresnet_best_ssim = srresnet_ssim
                srresnet_best_ssim_model = i + 1
            if srgan_ssim > srgan_best_ssim:
                srgan_best_ssim = srgan_ssim
                srgan_best_ssim_model = i + 1

    print('srcnn_best_PSNR:    {:.2f}   model:{}'.format(srcnn_best_psnr, srcnn_best_psnr_model))
    print('edsr_best_PSNR:    {:.2f}    model:{}'.format(edsr_best_psnr, edsr_best_psnr_model))
    print('srresnet_best_PSNR: {:.2f}   model:{}'.format(srresnet_best_psnr, srresnet_best_psnr_model))
    print('srgan_best_PSNR:    {:.2f}   model:{}'.format(srgan_best_psnr, srgan_best_psnr_model))
    print('asrresnet_best_PSNR: {:.2f}  model:{}'.format(asrresnet_best_psnr, asrresnet_best_psnr_model))
    print('asrgan_best_PSNR:   {:.2f}   model:{}'.format(asrgan_best_psnr, asrgan_best_psnr_model))
    print('sasrgan_best_PSNR:   {:.2f}   model:{}'.format(sasrgan_best_psnr, sasrgan_best_psnr_model))
    print('----------------------')

    print('srcnn_best_SSIM:    {:.4f}   model:{}'.format(srcnn_best_ssim, srcnn_best_ssim_model))
    print('edsr_best_SSIM:    {:.4f}    model:{}'.format(edsr_best_ssim, edsr_best_ssim_model))
    print('srresnet_best_SSIM: {:.4f}   model:{}'.format(srresnet_best_ssim, srresnet_best_ssim_model))
    print('srgan_best_SSIM:    {:.4f}   model:{}'.format(srgan_best_ssim, srgan_best_ssim_model))
    print('asrresnet_best_SSIM: {:.4f}  model:{}'.format(asrresnet_best_ssim, asrresnet_best_ssim_model))
    print('asrgan_best_SSIM:   {:.4f}   model:{}'.format(asrgan_best_ssim, asrgan_best_ssim_model))
    print('sasrgan_best_SSIM:   {:.4f}   model:{}'.format(sasrgan_best_ssim, sasrgan_best_ssim_model))

    print('----------------------')

    print(f'Finish')
