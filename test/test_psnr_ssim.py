
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
from data_utils import TestDatasetFromFolder, display_transform
from model.model_tsrgan import Generator_TSRGAN

if __name__ == '__main__':

    UPSCALE_FACTOR = 4

    TEST_DIR = '../data/test_x4'

    epoch_sum = 1

    psnr_set = []
    ssim_set = []

    test_set = TestDatasetFromFolder(TEST_DIR)
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
    test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

    for i in range(20):

        index = 1
        psnr_sum = 0
        ssim_sum = 0

        # MODEL_NAME = f'C:/code/train_results/model/t_x4/G/tsrgan_netG_epoch_4_{i+30}.pth'
        # MODEL_NAME = f'C:/code/train_results/new_model/tsrgan_x4/G/tsrgan_netG_epoch_4_{i+185}.pth'
        # MODEL_NAME = f'C:/code/SRCNN_Pytorch_1.0-master/SRCNN_Pytorch_1.0-master/outputs/x4/epoch_{i+1}.pth'
        MODEL_NAME = f'C:/code/train_results/new_model/tsrgan_v2_x4/G/tsrgan_v2_netG_epoch_4_{i+10}.pth'

        model = Generator_TSRGAN(UPSCALE_FACTOR).eval()
        if torch.cuda.is_available():
            model = model.cuda()
        model.load_state_dict(torch.load(MODEL_NAME), False)

        with torch.no_grad():

            for image_name, lr_image, hr_image in test_bar:

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
    data_frame.to_csv(out_path + 'tsrgan_v2_test_' + str(UPSCALE_FACTOR) + '.csv', index_label='Epoch')

    x = range(1, epoch_sum)

    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.plot(x, psnr_set)
    plt.xlabel("PSNR")

    plt.subplot(1, 2, 2)
    plt.plot(x, ssim_set)
    plt.xlabel("SSIM")

    plt.show()

