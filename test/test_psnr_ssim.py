
import os
from math import log10

import torch
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd

from tqdm import tqdm
import matplotlib.pyplot as plt
import pytorch_ssim
from data_utils import TestDatasetFromFolder, display_transform, ValDatasetFromFolder
from model.model_tsrgan import Generator_TSRGAN
from model.model_srgan import Generator
from model.model_srcnn import SRCNN
from model.model_asrgan import Generator_ASRGAN
from model.model_esdr import EDSR

if __name__ == '__main__':

    UPSCALE_FACTOR = 4

    TEST_DIR = f'../data/new_data/test_x{UPSCALE_FACTOR}'

    # MODEL = 'bicubic'
    # MODEL = 'edsr'
    MODEL = 'srcnn'
    # MODEL = 'srresnet'
    # MODEL = 'srgan'
    # MODEL = 'tsrgan'
    # MODEL = 'tsrgan_mse'
    # MODEL = 'tsrgan_v2'
    # MODEL = 'asrresnet'
    # MODEL = 'asrgan'

    epoch_sum = 1

    psnr_set = []
    ssim_set = []

    test_set = TestDatasetFromFolder(TEST_DIR)
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
    test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

    for i in range(100):

        index = 1
        psnr_sum = 0
        ssim_sum = 0

        if MODEL == 'srcnn' or MODEL == 'edsr':
            MODEL_NAME = f'C:/code/train_results/new_model/x{UPSCALE_FACTOR}/{MODEL}_x{UPSCALE_FACTOR}/model/{MODEL}_epoch_{UPSCALE_FACTOR}_{i + 1}.pth'
        else:
            MODEL_NAME = f'C:/code/train_results/new_model/x{UPSCALE_FACTOR}/{MODEL}_x{UPSCALE_FACTOR}/G/{MODEL}_netG_epoch_{UPSCALE_FACTOR}_{i + 1}.pth'

        if MODEL == 'srcnn':
            model = SRCNN().eval()
        if MODEL == 'edsr':
            model = EDSR(UPSCALE_FACTOR).eval()
        if MODEL == 'srgan':
            model = Generator(UPSCALE_FACTOR).eval()
        if MODEL == 'tsrgan':
            model = Generator_TSRGAN(UPSCALE_FACTOR).eval()
        if MODEL == 'asrgan':
            model = Generator_ASRGAN(UPSCALE_FACTOR).eval()

        if torch.cuda.is_available():
            model = model.cuda()
        model.load_state_dict(torch.load(MODEL_NAME), False)

        with torch.no_grad():

            for image_name, _, lr_image, hr_image in test_bar:

                image_name = image_name[0]
                lr_image = Variable(lr_image, volatile=True)
                hr_image = Variable(hr_image, volatile=True)

                if torch.cuda.is_available():
                    lr_image = lr_image.cuda()
                    hr_image = hr_image.cuda()

                sr_image = model(lr_image)

                psnr = 10 * log10(1 / ((hr_image - sr_image) ** 2).data.mean())
                # print('PSNR:  {:.4f}'.format(psnr))
                ssim = pytorch_ssim.ssim(sr_image, hr_image).item()
                # print('SSIM:  {:.4f}'.format(ssim))
                psnr_sum += psnr
                ssim_sum += ssim

                index += 1

            print(f'----{epoch_sum}----')
            print('PSNR:  {:.4f}'.format(psnr_sum / index))
            print('SSIM:  {:.4f}'.format(ssim_sum / index))

            psnr_set.append(psnr_sum / index)
            ssim_set.append(ssim_sum / index)

            epoch_sum += 1

    out_path = '../statistics/'
    data_frame = pd.DataFrame(
        data={'PSNR': psnr_set, 'SSIM': ssim_set},
        index=range(1, epoch_sum))
    data_frame.to_csv(out_path + f'{MODEL}_test_' + str(UPSCALE_FACTOR) + '.csv', index_label='Epoch')

    x = range(1, epoch_sum)

    plt.figure(1)
    plt.plot(x, psnr_set)
    plt.xlabel("PSNR")

    plt.figure(2)
    plt.plot(x, ssim_set)
    plt.xlabel("SSIM")

    plt.show()

