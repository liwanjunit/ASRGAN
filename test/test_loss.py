
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    UPSCALE_FACTOR = 2

    srcnn_path = f'C:/code/train_results/new_model/x{UPSCALE_FACTOR}/srcnn_x{UPSCALE_FACTOR}/srcnn_train_{UPSCALE_FACTOR}.csv'
    srresnet_path = f'C:/code/train_results/new_model/x{UPSCALE_FACTOR}/srresnet_x{UPSCALE_FACTOR}/srresnet_train_{UPSCALE_FACTOR}.csv'
    srgan_path = f'C:/code/train_results/new_model/x{UPSCALE_FACTOR}/srgan_x{UPSCALE_FACTOR}/srgan_train_{UPSCALE_FACTOR}.csv'
    tsrgan_path = f'C:/code/train_results/new_model/x{UPSCALE_FACTOR}/tsrgan_x{UPSCALE_FACTOR}/tsrgan_train_{UPSCALE_FACTOR}.csv'
    tsrgan_new_path = f'C:/code/train_results/new_model/x{UPSCALE_FACTOR}/tsrgan_new_x{UPSCALE_FACTOR}/tsrgan_new_train_{UPSCALE_FACTOR}.csv'
    tsrgan_pro_path = f'C:/code/train_results/new_model/x{UPSCALE_FACTOR}/tsrgan++_x{UPSCALE_FACTOR}/tsrgan++_train_{UPSCALE_FACTOR}.csv'
    tsrgan_mse_path = f'C:/code/train_results/new_model/x{UPSCALE_FACTOR}/tsrgan_mse_x{UPSCALE_FACTOR}/tsrgan_mse_train_{UPSCALE_FACTOR}.csv'
    tsrgan_v2_path = f'C:/code/train_results/new_model/x{UPSCALE_FACTOR}/tsrgan_v2_x{UPSCALE_FACTOR}/tsrgan_v2_train_{UPSCALE_FACTOR}.csv'
    asrgan_path = f'C:/code/train_results/new_model/x{UPSCALE_FACTOR}/asrgan_x{UPSCALE_FACTOR}/asrgan_train_{UPSCALE_FACTOR}.csv'


    srcnn_data = pd.read_csv(srcnn_path)
    # srresnet_data = pd.read_csv(srresnet_path)
    srgan_data = pd.read_csv(srgan_path)
    # tsrgan_data = pd.read_csv(tsrgan_path)
    # tsrgan_new_data = pd.read_csv(tsrgan_new_path)
    # tsrgan_pro_data = pd.read_csv(tsrgan_pro_path)
    # tsrgan_mse_data = pd.read_csv(tsrgan_mse_path)
    # tsrgan_v2_data = pd.read_csv(tsrgan_v2_path)
    # asrgan_data = pd.read_csv(asrgan_path)

    srcnn_loss_sum = []
    srresnet_loss_sum = []
    srgan_loss_sum = []
    tsrgan_loss_sum = []
    tsrgan_mse_loss_sum = []
    tsrgan_v2_losss_um = []
    asrgan_loss_sum = []

    for i in range(100):

        srcnn_loss_sum.append(srcnn_data['PSNR'][(i)])
        # srresnet_loss_sum.append(srresnet_data['SSIM'][(i)])
        srgan_loss_sum.append(srgan_data['SSIM'][(i)])
        # tsrgan_loss_sum.append(tsrgan_data['SSIM'][(i)])
        # tsrgan_mse_loss_sum.append(tsrgan_mse_data['SSIM'][(i)*40])
        # tsrgan_v2_losss_um.append(tsrgan_v2_data['PSNR'][(i)])
        # asrgan_loss_sum.append(tsrgan_v2_data['SSIM'][(i)])

    x = range(1, 100 + 1)

    plt.figure()
    plt.title('LOSS')
    # plt.plot(x, srcnn_loss_sum, color='g', linestyle='-', marker='', label='SRCNN')
    # plt.plot(x, srresnet_loss_sum, color='m', linestyle='-', label='SRResNet')
    # plt.plot(x, srgan_loss_sum, color='r', linestyle='-', marker='', label='SRGAN')
    # plt.plot(x, tsrgan_loss_sum, color='b', linestyle='-', marker='', label='TSRGAN')
    # plt.plot(x, tsrgan_mse_loss_sum, color='y', linestyle='-', label='TSRGAN_MSE')
    # plt.plot(x, tsrgan_v2_losss_um, color='y', linestyle='-', label='TSRGAN_v2')
    # plt.plot(x, asrgan_loss_sum, color='k', linestyle='-', marker='', label='BICUBIC')
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS')
    plt.legend(loc='center right')

    plt.show()


