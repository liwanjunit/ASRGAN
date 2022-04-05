
import os
from math import log10

import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TestDatasetFromFolder, display_transform
from model.model_tsrgan import Generator_TSRGAN

if __name__ == '__main__':

    UPSCALE_FACTOR = 4
    MODEL_NAME = 'tsrgan_netG_epoch_4_20.pth'

    results = {'data_17500': {'psnr': [], 'ssim': []}}

    model = Generator_TSRGAN(UPSCALE_FACTOR).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME), False)

    test_set = TestDatasetFromFolder('data/test', upscale_factor=UPSCALE_FACTOR)
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
    test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

    out_path = 'benchmark_results/SRF_' + str(UPSCALE_FACTOR) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for image_name, lr_image, hr_restore_img, hr_image, cnn_image, gan_image in test_bar:
        image_name = image_name[0]
        lr_image = Variable(lr_image, volatile=True)
        hr_image = Variable(hr_image, volatile=True)
        bic_image = Variable(hr_restore_img, volatile=True)
        cnn_image = Variable(cnn_image, volatile=True)
        gan_image = Variable(gan_image, volatile=True)
        if torch.cuda.is_available():
            lr_image = lr_image.cuda()
            hr_image = hr_image.cuda()
            bic_image = bic_image.cuda()
            cnn_image = cnn_image.cuda()
            gan_image = gan_image.cuda()

        sr_image = model(lr_image)

        bicubic_psnr = 10 * log10(1 / ((hr_image - bic_image) ** 2).data.mean())
        srcnn_psnr = 10 * log10(1 / ((hr_image - cnn_image) ** 2).data.mean())
        srgan_psnr = 10 * log10(1 / ((hr_image - gan_image) ** 2).data.mean())
        tsrgan_psnr = 10 * log10(1 / ((hr_image - sr_image) ** 2).data.mean())
        print('bicubic_PSNR: {:.4f}'.format(bicubic_psnr))
        print('srcnn_PSNR:   {:.4f}'.format(srcnn_psnr))
        print('srgan_PSNR:   {:.4f}'.format(srgan_psnr))
        print('tsrgan_PSNR:  {:.4f}'.format(tsrgan_psnr))


        bicubic_ssim = pytorch_ssim.ssim(bic_image, hr_image).item()
        srcnn_ssim = pytorch_ssim.ssim(cnn_image, hr_image).item()
        srgan_ssim = pytorch_ssim.ssim(gan_image, hr_image).item()
        tsrgan_ssim = pytorch_ssim.ssim(sr_image, hr_image).item()
        print('bicubic_SSIM: {:.4f}'.format(bicubic_ssim))
        print('srcnn_SSIM:   {:.4f}'.format(srcnn_ssim))
        print('srgan_SSIM:   {:.4f}'.format(srgan_ssim))
        print('tsrgan_SSIM:  {:.4f}'.format(tsrgan_ssim))


        test_images = torch.stack(
            [display_transform()(bic_image.data.cpu().squeeze(0)), display_transform()(cnn_image.data.cpu().squeeze(0)),
             display_transform()(gan_image.data.cpu().squeeze(0)), display_transform()(sr_image.data.cpu().squeeze(0)),
             display_transform()(hr_image.data.cpu().squeeze(0))])
        image = utils.make_grid(test_images, nrow=5, padding=5)
        utils.save_image(image, out_path + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (tsrgan_psnr, tsrgan_ssim) +
                         image_name.split('.')[-1], padding=5)

        # save psnr\ssim
        results['data_17500']['psnr'].append(tsrgan_psnr)
        results['data_17500']['ssim'].append(tsrgan_ssim)
        #
        # sr_image = ToPILImage()(sr_image[0].data.cpu())
        # sr_image.save('tsrgan_' + str(UPSCALE_FACTOR) + '_' + image_name + '.png')

    out_path = 'statistics/'
    saved_results = {'psnr': [], 'ssim': []}
    for item in results.values():
        psnr = np.array(item['psnr'])
        ssim = np.array(item['ssim'])
        if (len(psnr) == 0) or (len(ssim) == 0):
            psnr = 'No data'
            ssim = 'No data'
        else:
            psnr = psnr.mean()
            ssim = ssim.mean()
        saved_results['psnr'].append(psnr)
        saved_results['ssim'].append(ssim)

    data_frame = pd.DataFrame(saved_results, results.keys())
    data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_test_results.csv', index_label='DataSet')
