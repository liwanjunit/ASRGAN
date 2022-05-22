
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from model.model_srcnn import SRCNN


if __name__ == '__main__':

    CROP_SIZE = 128
    UPSCALE_FACTOR = 2
    NUM_EPOCHS = 100
    EPOCH_SUM = 0

    INIT_LR = 0.0001
    BATCH_SIZE = 2

    # MODEL_NAME = f'srcnn_epoch_{UPSCALE_FACTOR}_50.pth'

    print(f'crop_size:{CROP_SIZE}')
    print(f'epoch_sum:{EPOCH_SUM}')
    print(f'batch_size:{BATCH_SIZE}')
    print(f'upscale_factor:{UPSCALE_FACTOR}')

    # train_set = TrainDatasetFromFolder('/kaggle/input/data-14604/train', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    # val_set = ValDatasetFromFolder('/kaggle/input/data-14604/test', upscale_factor=UPSCALE_FACTOR)

    train_set = TrainDatasetFromFolder('../data_14604/train', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder('../data_14604/test', upscale_factor=UPSCALE_FACTOR)


    # train_set = TrainDatasetFromFolder('C:/code/SRGAN-master/VOC2012/VOC2012/train', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    # val_set = ValDatasetFromFolder('C:/code/SRGAN-master/VOC2012/VOC2012/val', upscale_factor=UPSCALE_FACTOR)

    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    model = SRCNN()
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))

    loss_function = nn.MSELoss()

    if torch.cuda.is_available():
        model.cuda()
        loss_function.cuda()
    #     model.load_state_dict(torch.load('epochs/' + MODEL_NAME), False)
    # else:
    #     model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))

    optimizerG = optim.Adam(model.parameters(), lr=INIT_LR)

    results = {'loss': [], 'psnr': [], 'ssim': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'loss': 0}

        model.train()

        for _, data, target in train_bar:

            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            hr = Variable(target)
            if torch.cuda.is_available():
                hr = hr.cuda()
            lr = Variable(data)
            if torch.cuda.is_available():
                lr = lr.cuda()

            model.zero_grad()
            sr = model(lr)
            loss = loss_function(sr, hr)
            loss.backward()
            optimizerG.step()

            running_results['loss'] += loss.item() * batch_size
            train_bar.set_description(desc='[%d/%d]Loss: %.4f ' % (epoch, NUM_EPOCHS, running_results['loss'] / running_results['batch_sizes']))

        model.eval()
        out_path = 'training_results/srcnn_SRF_' + str(UPSCALE_FACTOR) + '/'
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
                lr = val_hr_restore
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                sr = model(lr)

                if image_index == 300:
                    sr_image = ToPILImage()(sr[0].data.cpu())
                    sr_image.save('test_image/results/' + 'srcnn_%d_%d.png' % (UPSCALE_FACTOR, epoch + EPOCH_SUM))

                image_index += 1

                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10((hr.max() ** 2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))

            #     val_images.extend(
            #         [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
            #          display_transform()(sr.data.cpu().squeeze(0))])
            #
            # val_images = torch.stack(val_images)
            # val_images = torch.chunk(val_images, val_images.size(0) // 15)
            # val_save_bar = tqdm(val_images, desc='[saving training results]')
            # index = 1
            # for image in val_save_bar:
            #     image = utils.make_grid(image, nrow=3, padding=5)
            #     utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch + EPOCH_SUM, index), padding=5)
            #     index += 1
            #     break

        # save model parameters
        if epoch % 1 == 0 and epoch != 0:
            torch.save(model.state_dict(), 'epochs/srcnn_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch + EPOCH_SUM))

        # save loss\scores\psnr\ssim
        results['loss'].append(running_results['loss'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])

        if epoch % 1 == 0 and epoch != 0:
            out_path = 'statistics/'
            data_frame = pd.DataFrame(
                data={'Loss': results['loss'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'srcnn_train_' + str(UPSCALE_FACTOR) + f'_{epoch + EPOCH_SUM}.csv', index_label='Epoch')
