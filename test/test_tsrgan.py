
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    UPSCALE_FACTOR = 4

    # bicubic_path = f'C:/code/train_results/new_model/x{UPSCALE_FACTOR}/bicubic_test_{UPSCALE_FACTOR}.csv'
    # srcnn_path = f'C:/code/train_results/new_model/x{UPSCALE_FACTOR}/srcnn_x{UPSCALE_FACTOR}/srcnn_test_{UPSCALE_FACTOR}.csv'
    # srresnet_path = f'C:/code/train_results/new_model/x{UPSCALE_FACTOR}/srresnet_x{UPSCALE_FACTOR}/srresnet_test_{UPSCALE_FACTOR}.csv'
    srgan_path = f'C:/code/train_results/new_model/x{UPSCALE_FACTOR}/srgan_x{UPSCALE_FACTOR}/srgan_test_{UPSCALE_FACTOR}.csv'
    tsrgan_path = f'C:/code/train_results/new_model/x{UPSCALE_FACTOR}/tsrgan_x{UPSCALE_FACTOR}/tsrgan_test_{UPSCALE_FACTOR}.csv'
    # tsrgan_mse_path = f'C:/code/train_results/new_model/x{UPSCALE_FACTOR}/tsrgan_mse_x{UPSCALE_FACTOR}/tsrgan_mse_test_{UPSCALE_FACTOR}.csv'
    # tsrgan_v2_path = f'C:/code/train_results/new_model/x{UPSCALE_FACTOR}/tsrgan_v2_x{UPSCALE_FACTOR}/tsrgan_v2_test_{UPSCALE_FACTOR}.csv'
    tsrgan_new_path = f'C:/code/train_results/new_model/x{UPSCALE_FACTOR}/tsrgan_new_x{UPSCALE_FACTOR}/tsrgan_new_test_{UPSCALE_FACTOR}.csv'
    tsrgan_pro_path = f'C:/code/train_results/new_model/x{UPSCALE_FACTOR}/tsrgan++_x{UPSCALE_FACTOR}/tsrgan++_test_{UPSCALE_FACTOR}.csv'
    # asrgan_path = f'C:/code/train_results/new_model/x{UPSCALE_FACTOR}/asrgan_x{UPSCALE_FACTOR}/asrgan++_test_{UPSCALE_FACTOR}.csv'

    # bicubic_data = pd.read_csv(bicubic_path)
    # srcnn_data = pd.read_csv(srcnn_path)
    # srresnet_data = pd.read_csv(srresnet_path)
    srgan_data = pd.read_csv(srgan_path)
    tsrgan_data = pd.read_csv(tsrgan_path)
    # tsrgan_mse_data = pd.read_csv(tsrgan_mse_path)
    # tsrgan_v2_data = pd.read_csv(tsrgan_v2_path)
    tsrgan_new_data = pd.read_csv(tsrgan_new_path)
    tsrgan_pro_data = pd.read_csv(tsrgan_pro_path)
    # asrgan_data = pd.read_csv(asrgan_path)

    # bicubic_psnr_sum = []
    # bicubic_ssim_sum = []
    #
    # srcnn_psnr_sum = []
    # srcnn_ssim_sum = []
    #
    # srresnet_psnr_sum = []
    # srresnet_ssim_sum = []

    srgan_psnr_sum = []
    srgan_ssim_sum = []

    tsrgan_psnr_sum = []
    tsrgan_ssim_sum = []

    # tsrgan_mse_psnr_sum = []
    # tsrgan_mse_ssim_sum = []
    #
    # tsrgan_v2_psnr_sum = []
    # tsrgan_v2_ssim_sum = []

    tsrgan_new_psnr_sum = []
    tsrgan_new_ssim_sum = []

    tsrgan_pro_psnr_sum = []
    tsrgan_pro_ssim_sum = []

    asrgan_psnr_sum = []
    asrgan_ssim_sum = []


    for i in range(90):

        # bicubic_psnr_sum.append(bicubic_data['PSNR'][(i)])
        # bicubic_ssim_sum.append(bicubic_data['SSIM'][(i)])
        #
        # srcnn_psnr_sum.append(srcnn_data['PSNR'][(i)])
        # srcnn_ssim_sum.append(srcnn_data['SSIM'][(i)])

        # srresnet_psnr_sum.append(srresnet_data['PSNR'][(i)])
        # srresnet_ssim_sum.append(srresnet_data['SSIM'][(i)])

        srgan_psnr_sum.append(srgan_data['PSNR'][(i+10)])
        srgan_ssim_sum.append(srgan_data['SSIM'][(i+10)])

        tsrgan_psnr_sum.append(tsrgan_data['PSNR'][(i+10)])
        tsrgan_ssim_sum.append(tsrgan_data['SSIM'][(i+10)])

        # tsrgan_mse_psnr_sum.append(tsrgan_mse_data['PSNR'][(i)*40])
        # tsrgan_mse_ssim_sum.append(tsrgan_mse_data['SSIM'][(i)*40])

        # tsrgan_v2_psnr_sum.append(tsrgan_v2_data['PSNR'][(i)])
        # tsrgan_v2_ssim_sum.append(tsrgan_v2_data['SSIM'][(i)])

        tsrgan_new_psnr_sum.append(tsrgan_new_data['PSNR'][i+10])
        tsrgan_new_ssim_sum.append(tsrgan_new_data['SSIM'][i+10])

        tsrgan_pro_psnr_sum.append(tsrgan_pro_data['PSNR'][i+10])
        tsrgan_pro_ssim_sum.append(tsrgan_pro_data['SSIM'][i+10])

        # asrgan_psnr_sum.append(asrgan_data['PSNR'][i])
        # asrgan_ssim_sum.append(asrgan_data['SSIM'][i])

    x = range(1, 90 + 1)

    plt.figure(1)
    # plt.subplot(1, 2, 1)
    plt.title('PSNR')
    # plt.plot(x, bicubic_psnr_sum, color='k', linestyle='--', marker='', label='BICUBIC')
    # plt.plot(x, srcnn_psnr_sum, color='g', linestyle=':', marker='', label='SRCNN')
    # plt.plot(x, srresnet_psnr_sum, color='m', linestyle='--', label='SRResNet')
    plt.plot(x, srgan_psnr_sum, color='r', linestyle='-.', marker='', label='SRGAN')
    plt.plot(x, tsrgan_psnr_sum, color='b', linestyle='-', marker='', label='TSRGAN')
    # plt.plot(x, tsrgan_mse_psnr_sum, color='y', linestyle='-', label='TSRGAN_MSE')
    # plt.plot(x, tsrgan_v2_psnr_sum, color='y', linestyle='-', label='TSRGAN_v2')
    plt.plot(x, tsrgan_new_psnr_sum, color='y', linestyle='-', label='TSRGAN_NEW')
    plt.plot(x, tsrgan_pro_psnr_sum, color='g', linestyle='-', label='TSRGAN++')
    # plt.plot(x, asrgan_psnr_sum, color='b', linestyle='-', marker='', label='ASRGAN')
    plt.xlabel('EPOCH')
    plt.ylabel('PSNR(dB)')
    plt.legend(loc='center right')

    plt.figure(2)
    # plt.subplot(1, 2, 2)
    plt.title('SSIM')
    # plt.plot(x, bicubic_ssim_sum, color='k', linestyle='--', marker='', label='BICUBIC')
    # plt.plot(x, srcnn_ssim_sum, color='g', linestyle=':',  marker='', label='SRCNN')
    # plt.plot(x, srresnet_ssim_sum, color='m', linestyle='--', label='SRResNet')
    plt.plot(x, srgan_ssim_sum, color='r', linestyle='-.',  marker='', label='SRGAN')
    plt.plot(x, tsrgan_ssim_sum, color='b', linestyle='-',  marker='', label='TSRGAN')
    # plt.plot(x, tsrgan_mse_ssim_sum, color='y', linestyle='-', label='TSRGAN_MSE')
    # plt.plot(x, tsrgan_v2_ssim_sum, color='y', linestyle='-', label='TSRGAN_v2')
    plt.plot(x, tsrgan_new_ssim_sum, color='y', linestyle='-', label='TSRGAN_NEW')
    plt.plot(x, tsrgan_pro_ssim_sum, color='g', linestyle='-', label='TSRGAN++')
    # plt.plot(x, asrgan_ssim_sum, color='b', linestyle='',  marker='', label='ASRGAN')

    plt.xlabel('EPOCH')
    plt.ylabel('SSIM')
    # plt.ylim(ymin=0.3, ymax=0.95)
    plt.legend(loc='center right')

    plt.show()
