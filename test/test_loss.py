
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    UPSCALE_FACTOR = 4

    srcnn_path = f'E:/code/train_results/new_model/x{UPSCALE_FACTOR}/srcnn_x{UPSCALE_FACTOR}/srcnn_train_{UPSCALE_FACTOR}.csv'
    srresnet_path = f'E:/code/train_results/new_model/x{UPSCALE_FACTOR}/srresnet_x{UPSCALE_FACTOR}/srresnet_train_{UPSCALE_FACTOR}.csv'
    srgan_path = f'E:/code/train_results/new_model/x{UPSCALE_FACTOR}/srgan_x{UPSCALE_FACTOR}/srgan_train_{UPSCALE_FACTOR}.csv'
    asrgan_path = f'E:/code/train_results/new_model/x{UPSCALE_FACTOR}/asrgan_x{UPSCALE_FACTOR}/asrgan_train_{UPSCALE_FACTOR}.csv'


    srcnn_data = pd.read_csv(srcnn_path)
    srresnet_data = pd.read_csv(srresnet_path)
    srgan_data = pd.read_csv(srgan_path)
    asrgan_data = pd.read_csv(asrgan_path)

    srcnn_loss_sum = []
    srresnet_loss_sum = []
    srgan_loss_sum = []
    asrgan_loss_sum = []

    for i in range(125):

        # srcnn_loss_sum.append(srcnn_data['PSNR'][(i)])
        # srresnet_loss_sum.append(srresnet_data['SSIM'][(i)])
        # srgan_loss_sum.append(srgan_data['SSIM'][(i)])
        asrgan_loss_sum.append(asrgan_data['SSIM'][(i)])

    x = range(1, 125 + 1)

    plt.figure()
    plt.title('LOSS')
    # plt.plot(x, srcnn_loss_sum, color='g', linestyle='-', marker='', label=f'SRCNN_x{UPSCALE_FACTOR}')
    # plt.plot(x, srresnet_loss_sum, color='m', linestyle='-', label=f'SRResNet_x{UPSCALE_FACTOR}')
    # plt.plot(x, srgan_loss_sum, color='r', linestyle='-', marker='', label=f'SRGAN_x{UPSCALE_FACTOR}')
    plt.plot(x, asrgan_loss_sum, color='k', linestyle='-', marker='', label=f'ASRGAN_x{UPSCALE_FACTOR}')
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS')
    plt.legend(loc='center right')

    plt.show()


