
import torch
from PIL import Image, ImageDraw
import pytorch_ssim
import torchvision.utils as utils
from torchvision.transforms import ToTensor, ToPILImage, Resize
from math import log10
from torch.autograd import Variable
from data_utils import display_transform
import matplotlib.pyplot as plt
from model.model_tsrgan import Generator_TSRGAN
from model.model_srcnn import SRCNN
from model.model_srgan import Generator
from model.model_asrgan import Generator_ASRGAN
from model.model_esdr import EDSR

if __name__ == '__main__':

    xmin = 1095
    xmax = 1351
    ymin = 595
    ymax = 851
    box = (xmin, ymin, xmax, ymax)

    CROP_SIZE = 256
    UPSCALE_FACTOR = 4

    image_name = 'M51.jpg'
    print('image_name: ' + image_name)
    print('----------------------')

    PATH = 'E:/code/ASRGAN/ASRGAN-master/test/compared/'

    SRCNN_MODEL_NAME = f'E:/code/train_results/new_model/x{UPSCALE_FACTOR}/srcnn_x{UPSCALE_FACTOR}/model/srcnn_epoch_{UPSCALE_FACTOR}_69.pth'
    EDSR_MODEL_NAME = f'E:/code/train_results/new_model/x{UPSCALE_FACTOR}/edsr_x{UPSCALE_FACTOR}/model/edsr_epoch_{UPSCALE_FACTOR}_95.pth'
    SRResNet_MODEL_NAME = f'E:/code/train_results/new_model/x{UPSCALE_FACTOR}/srresnet_x{UPSCALE_FACTOR}/model/srresnet_epoch_{UPSCALE_FACTOR}_85.pth'
    SRGAN_MODEL_NAME = f'E:/code/train_results/new_model/x{UPSCALE_FACTOR}/srgan_x{UPSCALE_FACTOR}/G/srgan_netG_epoch_{UPSCALE_FACTOR}_98.pth'
    ASRResNet_MODEL_NAME = f'E:/code/train_results/new_model/x{UPSCALE_FACTOR}/asrresnet_x{UPSCALE_FACTOR}/model/asrresnet_epoch_{UPSCALE_FACTOR}_94.pth'
    ASRGAN_MODEL_NAME = f'E:/code/train_results/new_model/x{UPSCALE_FACTOR}/asrgan_x{UPSCALE_FACTOR}/G/asrgan_netG_epoch_{UPSCALE_FACTOR}_89.pth'

    # hr_path = PATH + 'test/'
    hr_path = PATH + 'new/'

    srcnn_model = SRCNN().eval()
    edsr_model = EDSR(UPSCALE_FACTOR).eval()
    srresnet_model = Generator(UPSCALE_FACTOR).eval()
    srgan_model = Generator(UPSCALE_FACTOR).eval()
    asrresnet_model = Generator_ASRGAN(UPSCALE_FACTOR).eval()
    asrgan_model = Generator_ASRGAN(UPSCALE_FACTOR).eval()

    if torch.cuda.is_available():
        srcnn_model = srcnn_model.cuda()
        edsr_model = edsr_model.cuda()
        srresnet_model = srresnet_model.cuda()
        srgan_model = srgan_model.cuda()
        asrresnet_model = asrresnet_model.cuda()
        asrgan_model = asrgan_model.cuda()

    srcnn_model.load_state_dict(torch.load(SRCNN_MODEL_NAME), False)
    edsr_model.load_state_dict(torch.load(EDSR_MODEL_NAME), False)
    srresnet_model.load_state_dict(torch.load(SRResNet_MODEL_NAME), False)
    srgan_model.load_state_dict(torch.load(SRGAN_MODEL_NAME), False)
    asrresnet_model.load_state_dict(torch.load(ASRResNet_MODEL_NAME), False)
    asrgan_model.load_state_dict(torch.load(ASRGAN_MODEL_NAME), False)

    img = Image.open(hr_path + image_name)
    hr_image = img.crop(box)

    lr_scale = Resize(CROP_SIZE // UPSCALE_FACTOR, interpolation=Image.BICUBIC)
    bicubic_scale = Resize(CROP_SIZE, interpolation=Image.BICUBIC)
    lr_image = lr_scale(hr_image)
    bicubic_image = bicubic_scale(lr_image)

    with torch.no_grad():

        lr_image = Variable(ToTensor()(lr_image), volatile=True).unsqueeze(0)
        hr_image = Variable(ToTensor()(hr_image), volatile=True).unsqueeze(0)
        bicubic_image = Variable(ToTensor()(bicubic_image), volatile=True).unsqueeze(0)

        if torch.cuda.is_available():
            lr_image = lr_image.cuda()
            hr_image = hr_image.cuda()
            bicubic_image = bicubic_image.cuda()

        srcnn_image = srcnn_model(bicubic_image)
        edsr_image = edsr_model(lr_image)
        srresnet_image = srresnet_model(lr_image)
        srgan_image = srgan_model(lr_image)
        asrresnet_image = asrresnet_model(lr_image)
        asrgan_image = asrgan_model(lr_image)

        bicubic_psnr = 10 * log10(1 / ((hr_image - bicubic_image) ** 2).data.mean())
        srcnn_psnr = 10 * log10(1 / ((hr_image - srcnn_image) ** 2).data.mean())
        edsr_psnr = 10 * log10(1 / ((hr_image - edsr_image) ** 2).data.mean())
        srresnet_psnr = 10 * log10(1 / ((hr_image - srresnet_image) ** 2).data.mean())
        srgan_psnr = 10 * log10(1 / ((hr_image - srgan_image) ** 2).data.mean())
        asrresnet_psnr = 10 * log10(1 / ((hr_image - asrresnet_image) ** 2).data.mean())
        asrgan_psnr = 10 * log10(1 / ((hr_image - asrgan_image) ** 2).data.mean())
        print('bicubic_PSNR:  {:.2f}'.format(bicubic_psnr))
        print('srcnn_PSNR:    {:.2f}'.format(srcnn_psnr))
        print('edsr_PSNR:    {:.2f}'.format(edsr_psnr))
        print('srresnet_PSNR: {:.2f}'.format(srresnet_psnr))
        print('srgan_PSNR:    {:.2f}'.format(srgan_psnr))
        print('asrresnet_PSNR: {:.2f}'.format(asrresnet_psnr))
        print('asrgan_PSNR:   {:.2f}'.format(asrgan_psnr))

        print('----------------------')

        bicubic_ssim = pytorch_ssim.ssim(bicubic_image, hr_image)
        srcnn_ssim = pytorch_ssim.ssim(srcnn_image, hr_image)
        edsr_ssim = pytorch_ssim.ssim(edsr_image, hr_image)
        srresnet_ssim = pytorch_ssim.ssim(srresnet_image, hr_image)
        srgan_ssim = pytorch_ssim.ssim(srgan_image, hr_image).item()
        asrresnet_ssim = pytorch_ssim.ssim(asrresnet_image, hr_image)
        asrgan_ssim = pytorch_ssim.ssim(asrgan_image, hr_image).item()

        print('bicubic_SSIM:  {:.4f}'.format(bicubic_ssim))
        print('srcnn_SSIM:    {:.4f}'.format(srcnn_ssim))
        print('edsr_SSIM:    {:.4f}'.format(edsr_ssim))
        print('srresnet_SSIM: {:.4f}'.format(srresnet_ssim))
        print('srgan_SSIM:    {:.4f}'.format(srgan_ssim))
        print('asrresnet_SSIM: {:.4f}'.format(asrresnet_ssim))
        print('asrgan_SSIM:   {:.4f}'.format(asrgan_ssim))

        print('----------------------')

    test_images = torch.stack(
        [
         display_transform()(hr_image.data.cpu().squeeze(0)),
         display_transform()(bicubic_image.data.cpu().squeeze(0)),
         display_transform()(srcnn_image.data.cpu().squeeze(0)),
         display_transform()(edsr_image.data.cpu().squeeze(0)),
         display_transform()(srresnet_image.data.cpu().squeeze(0)),
         display_transform()(srgan_image.data.cpu().squeeze(0)),
         display_transform()(asrresnet_image.data.cpu().squeeze(0)),
         display_transform()(asrgan_image.data.cpu().squeeze(0)),
         ])
    image = utils.make_grid(test_images, nrow=4, padding=5)
    utils.save_image(image, PATH + image_name.split('.')[0] + '_compared' + '.png')

    hr_image = ToPILImage()(hr_image[0].data.cpu())
    bicubic_image = ToPILImage()(bicubic_image[0].data.cpu())
    srcnn_image = ToPILImage()(srcnn_image[0].data.cpu())
    edsr_image = ToPILImage()(edsr_image[0].data.cpu())
    srresnet_image = ToPILImage()(srresnet_image[0].data.cpu())
    srgan_image = ToPILImage()(srgan_image[0].data.cpu())
    asrresnet_image = ToPILImage()(asrresnet_image[0].data.cpu())
    asrgan_image = ToPILImage()(asrgan_image[0].data.cpu())

    hr_image.save(PATH + 'results/' + image_name.split('.')[0] + '_hr' + f'_{box}' + '.png')
    bicubic_image.save(PATH + 'results/' + image_name.split('.')[0] + '_bicubic' + f'_psnr_{bicubic_psnr}_ssim_{bicubic_ssim}' + '.png')
    srcnn_image.save(PATH + 'results/' + image_name.split('.')[0] + '_srcnn' + f'_psnr_{srcnn_psnr}_ssim_{srcnn_ssim}' + '.png')
    edsr_image.save(PATH + 'results/' + image_name.split('.')[0] + '_edsr' + f'_psnr_{edsr_psnr}_ssim_{edsr_ssim}' + '.png')
    srresnet_image.save(PATH + 'results/' + image_name.split('.')[0] + '_srresnet' + f'_psnr_{srresnet_psnr}_ssim_{srresnet_ssim}' + '.png')
    srgan_image.save(PATH + 'results/' + image_name.split('.')[0] + '_srgan' + f'_psnr_{srgan_psnr}_ssim_{srgan_ssim}' + '.png')
    asrresnet_image.save(PATH + 'results/' + image_name.split('.')[0] + '_asrresnet' + f'_psnr_{asrresnet_psnr}_ssim_{asrresnet_ssim}' + '.png')
    asrgan_image.save(PATH + 'results/' + image_name.split('.')[0] + '_asrgan' + f'_psnr_{asrgan_psnr}_ssim_{asrgan_ssim}' + '.png')

    result = Image.open(PATH + image_name.split('.')[0] + '_compared' + '.png')

    draw = ImageDraw.Draw(img)
    draw.rectangle([xmin, ymin, xmax, ymax], outline='yellow', width=10)
    img.save(PATH + image_name.split('.')[0] + '_draw' + '.png')

    plt.figure()
    plt.imshow(result)
    # plt.show()

    print(f'Finish')
