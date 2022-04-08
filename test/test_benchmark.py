
from math import log10
import pandas as pd
import torch

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from tqdm import tqdm

import pytorch_ssim
from data_utils import TestDatasetFromFolder
from model.model_tsrgan import Generator_TSRGAN
from model.model_srcnn import SRCNN
from model.model import Generator

if __name__ == '__main__':

    UPSCALE_FACTOR = 4

    # MODEL_NAME = 'C:/code/SRCNN_Pytorch_1.0-master/SRCNN_Pytorch_1.0-master/outputs/x4/epoch_144.pth'
    # MODEL_NAME = 'C:/code/train_results/new_model/srgan_x4/G/srgan_netG_epoch_4_193.pth'
    MODEL_NAME = 'C:/code/train_results/new_model/tsrgan_x4/G/tsrgan_netG_epoch_4_145.pth'

    dataset_dir = '../data/test_x4'

    # image_out_path = 'C:/code/ASRGAN/ASRGAN-master/test_image/compared/srcnn/'
    # image_out_path = 'C:/code/ASRGAN/ASRGAN-master/test_image/compared/srgan/'
    image_out_path = 'C:/code/ASRGAN/ASRGAN-master/test_image/compared/tsrgan/'

    image_name_set = []
    psnr_set = []
    ssim_set = []

    index = 1

    # model = SRCNN().eval()
    # model = Generator(UPSCALE_FACTOR).eval()
    model = Generator_TSRGAN(UPSCALE_FACTOR).eval()

    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(MODEL_NAME), False)

    test_set = TestDatasetFromFolder(dataset_dir)
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
    test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

    for image_name, lr_image, hr_image in test_bar:

        image_name = image_name[0]
        lr_image = Variable(lr_image, volatile=True)
        hr_image = Variable(hr_image, volatile=True)

        if torch.cuda.is_available():
            lr_image = lr_image.cuda()
            hr_image = hr_image.cuda()

        sr_image = model(lr_image)

        psnr = 10 * log10(1 / ((hr_image - sr_image) ** 2).data.mean())
        ssim = pytorch_ssim.ssim(sr_image, hr_image).item()

        image_name_set.append(image_name)
        psnr_set.append(psnr)
        ssim_set.append(ssim)

        sr_image = ToPILImage()(sr_image[0].data.cpu())
        sr_image.save(image_out_path + image_name)

        index += 1

    out_path = '../statistics/'
    data_frame = pd.DataFrame(
        data={'Image_name': image_name_set, 'PSNR': psnr_set, 'SSIM': ssim_set},
        index=range(1, index))

    # data_frame.to_csv(out_path + 'srcnn_image_' + str(UPSCALE_FACTOR) + '.csv', index_label='Index')
    # data_frame.to_csv(out_path + 'srgan_image_' + str(UPSCALE_FACTOR) + '.csv', index_label='Index')
    data_frame.to_csv(out_path + 'tsrgan_image_' + str(UPSCALE_FACTOR) + '.csv', index_label='Index')

    print('Finish')
