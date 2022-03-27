
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from model import Generator
from model_transformer import ViT


if __name__ == '__main__':

    CROP_SIZE = 128
    UPSCALE_FACTOR = 4
    NUM_EPOCHS = 20
    EPOCH_SUM = 30
    BATCH_SIZE = 2

    D_INIT_LR = 0.0001
    G_INIT_LR = 0.0001

    MODEL_NAME_G = '/kaggle/input/sr-model/tsrgan_v2_netG_epoch_4_30.pth'
    MODEL_NAME_D = '/kaggle/input/sr-model/tsrgan_v2_netD_epoch_4_30.pth'

    print(f'epoch_sum:{EPOCH_SUM}')
    print(f'batch_size:{BATCH_SIZE}')
    print(f'upscale_factor:{UPSCALE_FACTOR}')

    train_set = TrainDatasetFromFolder('/kaggle/input/data-14000/train', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder('/kaggle/input/data-14000/val', upscale_factor=UPSCALE_FACTOR)
    #
    # train_set = TrainDatasetFromFolder('../data_14000/train', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    # val_set = ValDatasetFromFolder('../data_14000/val', upscale_factor=UPSCALE_FACTOR)

    # train_set = TrainDatasetFromFolder('C:/code/SRGAN-master/VOC2012/VOC2012/train', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    # val_set = ValDatasetFromFolder('C:/code/SRGAN-master/VOC2012/VOC2012/val', upscale_factor=UPSCALE_FACTOR)

    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    netG = Generator(UPSCALE_FACTOR)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = ViT(
        image_size=128,
        patch_size=16,
        num_classes=1,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    generator_criterion = GeneratorLoss()

    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()
        # netG.load_state_dict(torch.load('epochs/' + MODEL_NAME_G), False)
        # netD.load_state_dict(torch.load('epochs/' + MODEL_NAME_D), False)
        netG.load_state_dict(torch.load(MODEL_NAME_G), False)
        netD.load_state_dict(torch.load(MODEL_NAME_D), False)
    else:
        netG.load_state_dict(torch.load(MODEL_NAME_G, map_location=lambda storage, loc: storage))
        netD.load_state_dict(torch.load(MODEL_NAME_D, map_location=lambda storage, loc: storage))
        # netG.load_state_dict(torch.load('epochs/' + MODEL_NAME_G, map_location=lambda storage, loc: storage))
        # netD.load_state_dict(torch.load('epochs/' + MODEL_NAME_D, map_location=lambda storage, loc: storage))

    optimizerG = optim.Adam(netG.parameters(), lr=G_INIT_LR)
    optimizerD = optim.Adam(netD.parameters(), lr=D_INIT_LR)

    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()
        for data, target in train_bar:
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()
            fake_img = netG(z)

            netD.zero_grad()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            ## The two lines below are added to prevent runetime error in Google Colab ##
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()
            ##
            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()

            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            optimizerG.step()

            # loss for current batch before optimization
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        netG.eval()
        out_path = 'training_results/tsrgan_v2_SRF_' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            image_index = 1
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                sr = netG(lr)

                if image_index == 1:
                    sr_image = ToPILImage()(sr[0].data.cpu())
                    sr_image.save('test_image/results/' + 'tsrgan_v2_%d_%d.png' % (UPSCALE_FACTOR, epoch + EPOCH_SUM))

                image_index += 1

                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10(
                    (hr.max() ** 2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))

                val_images.extend(
                    [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0))])
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + 'tsrgan_v2_epoch_%d_index_%d.png' % (epoch + EPOCH_SUM, index), padding=5)
                index += 1

        # save model parameters
        if epoch % 1 == 0 and epoch != 0:
            torch.save(netG.state_dict(), 'epochs/tsrgan_v2_netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch + EPOCH_SUM))
            torch.save(netD.state_dict(), 'epochs/tsrgan_v2_netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch + EPOCH_SUM))

        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])

        if epoch % 1 == 0 and epoch != 0:
            out_path = 'statistics/'
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'tsrgan_v2_srf_' + str(UPSCALE_FACTOR) + F'_{EPOCH_SUM + epoch}.csv', index_label='Epoch')
