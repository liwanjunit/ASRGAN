import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import ValDatasetFromFolder, display_transform
from model import Generator
import time
import matplotlib.pyplot as plt
import torchvision.utils as utils
from torchvision.transforms import Compose, ToTensor, ToPILImage, CenterCrop, Resize
from math import log10


def calc_psnr(img1, img2):
    return 10. * log10(img2.max() ** 2 / torch.mean((img1 / 1. - img2 / 1.) ** 2))


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    out_path = 'test_image/results_set/'
    epoch = 100

    val_set = ValDatasetFromFolder('../data_14000/val', upscale_factor=4)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    G = Generator(4).to(device)
    G.load_state_dict(torch.load(f'C:/Users/lai/Downloads/netG_epoch_4_{epoch}.pth', map_location=device))

    val_bar = tqdm(val_loader)
    val_images = []
    index = 1

    for val_lr, val_hr_restore, val_hr in val_bar:

        batch_size = val_lr.size(0)

        lr = val_lr
        hr = val_hr
        if torch.cuda.is_available():
            lr = lr.cuda()
            hr = hr.cuda()
        sr = G(lr)

        # val_images.extend(
        #     [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
        #      display_transform()(sr.data.cpu().squeeze(0))])

        print(f'---------第{index}个---------')
        sr_psnr = calc_psnr(hr, sr)
        print('sr_PSNR: {:.2f}'.format(sr_psnr))
        sr_ssim = pytorch_ssim.ssim(sr, hr)
        print('sr_SSIM: {:.2f}'.format(sr_ssim))

        sr_image = ToPILImage()(sr[0].data.cpu())
        sr_image.save(out_path + 'srgan_%d_%d.png' % (4, index))
        index += 1

    # val_images = torch.stack(val_images)
    # val_images = torch.chunk(val_images, val_images.size(0) // 15)
    # val_save_bar = tqdm(val_images, desc='[saving training results]')
    # index = 1
    #
    # for image in val_save_bar:
    #     image = utils.make_grid(image, nrow=3, padding=5)
    #     utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
    #     index += 1

    print(f'Finish')


