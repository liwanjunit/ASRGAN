
import os
from math import log10

import torch
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tqdm import tqdm
import matplotlib.pyplot as plt
import pytorch_ssim
from data_utils import TestDatasetFromFolder, display_transform
from model.model_srgan import Generator

if __name__ == '__main__':

    UPSCALE_FACTOR = 4
    index = 1

    MODEL_NAME = 'srgan_netG_epoch_4_50.pth'
    TEST_PATH = 'data/test'

    psnr_set = []
    ssim_set = []

    max_psnr = 0
    max_psnr_index = 0
    max_ssim = 0
    max_ssim_index = 0

    model = Generator(UPSCALE_FACTOR).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME), False)

    test_set = TestDatasetFromFolder(TEST_PATH, upscale_factor=UPSCALE_FACTOR)
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
    test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

    out_path = 'benchmark_results/SRF_' + str(UPSCALE_FACTOR) + '/'

    if not os.path.exists(out_path):
        os.makedirs(out_path)

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

        test_images = torch.stack([display_transform()(sr_image.data.cpu().squeeze(0)), display_transform()(hr_image.data.cpu().squeeze(0))])
        image = utils.make_grid(test_images, nrow=2, padding=5)
        utils.save_image(image, out_path + f'{index}' + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) +
                         image_name.split('.')[-1], padding=5)


        if psnr > max_psnr:
            max_psnr = psnr
            max_psnr_index = index
        if ssim > max_ssim:
            max_ssim = ssim
            max_ssim_index = index

        psnr_set.append(psnr)
        ssim_set.append(ssim)

        index += 1

    print('max_psnr：{:.4f}'.format(max_psnr))
    print('index：{:.4f}' .format(max_psnr_index))
    print('max_ssim：{:.4f}'.format(max_ssim))
    print('index：{:.4f}' .format(max_ssim_index))

    x = range(1, index)
    plt.figure()
    plt.plot(x, psnr_set)
    plt.show()

