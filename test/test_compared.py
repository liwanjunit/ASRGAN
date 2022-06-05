
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    UPSCALE_FACTOR = 8

    bilinear_path = f'E:/code/train_results/new_model/x{UPSCALE_FACTOR}/bilinear_test_{UPSCALE_FACTOR}.csv'
    bicubic_path = f'E:/code/train_results/new_model/x{UPSCALE_FACTOR}/bicubic_test_{UPSCALE_FACTOR}.csv'
    srcnn_path = f'E:/code/train_results/new_model/x{UPSCALE_FACTOR}/srcnn_x{UPSCALE_FACTOR}/srcnn_test_{UPSCALE_FACTOR}.csv'
    # edsr_path = f'E:/code/train_results/new_model/x{UPSCALE_FACTOR}/edsr_x{UPSCALE_FACTOR}/edsr_test_{UPSCALE_FACTOR}.csv'
    srresnet_path = f'E:/code/train_results/new_model/x{UPSCALE_FACTOR}/srresnet_x{UPSCALE_FACTOR}/srresnet_test_{UPSCALE_FACTOR}.csv'
    srgan_path = f'E:/code/train_results/new_model/x{UPSCALE_FACTOR}/srgan_x{UPSCALE_FACTOR}/srgan_test_{UPSCALE_FACTOR}.csv'
    asrresnet_path = f'E:/code/train_results/new_model/x{UPSCALE_FACTOR}/asrresnet_x{UPSCALE_FACTOR}/asrresnet_test_{UPSCALE_FACTOR}.csv'
    asrgan_path = f'E:/code/train_results/new_model/x{UPSCALE_FACTOR}/asrgan_x{UPSCALE_FACTOR}/asrgan_test_{UPSCALE_FACTOR}.csv'

    bilinear_data = pd.read_csv(bilinear_path)
    bicubic_data = pd.read_csv(bicubic_path)
    srcnn_data = pd.read_csv(srcnn_path)
    # edsr_data = pd.read_csv(edsr_path)
    srresnet_data = pd.read_csv(srresnet_path)
    srgan_data = pd.read_csv(srgan_path)
    asrresnet_data = pd.read_csv(asrresnet_path)
    asrgan_data = pd.read_csv(asrgan_path)

    bilinear_psnr_sum = []
    bilinear_ssim_sum = []

    bicubic_psnr_sum = []
    bicubic_ssim_sum = []

    srcnn_psnr_sum = []
    srcnn_ssim_sum = []

    # edsr_psnr_sum = []
    # edsr_ssim_sum = []

    srresnet_psnr_sum = []
    srresnet_ssim_sum = []

    srgan_psnr_sum = []
    srgan_ssim_sum = []

    asrresnet_psnr_sum = []
    asrresnet_ssim_sum = []

    asrgan_psnr_sum = []
    asrgan_ssim_sum = []

    k = 0

    for i in range(100-k):

        bilinear_psnr_sum.append(bilinear_data['PSNR'][i + k])
        bilinear_ssim_sum.append(bilinear_data['SSIM'][i + k])

        bicubic_psnr_sum.append(bicubic_data['PSNR'][i + k])
        bicubic_ssim_sum.append(bicubic_data['SSIM'][i + k])

        srcnn_psnr_sum.append(srcnn_data['PSNR'][i + k])
        srcnn_ssim_sum.append(srcnn_data['SSIM'][i + k])

        # edsr_psnr_sum.append(edsr_data['PSNR'][i + k])
        # edsr_ssim_sum.append(edsr_data['SSIM'][i + k])

        srresnet_psnr_sum.append(srresnet_data['PSNR'][i + k])
        srresnet_ssim_sum.append(srresnet_data['SSIM'][i + k])

        srgan_psnr_sum.append(srgan_data['PSNR'][i + k])
        srgan_ssim_sum.append(srgan_data['SSIM'][i + k])

        asrresnet_psnr_sum.append(asrresnet_data['PSNR'][i + k])
        asrresnet_ssim_sum.append(asrresnet_data['SSIM'][i + k])

        asrgan_psnr_sum.append(asrgan_data['PSNR'][i + k])
        asrgan_ssim_sum.append(asrgan_data['SSIM'][i + k])

    x = range(1, 100-k + 1)

    plt.figure(1)
    # plt.subplot(1, 2, 1)
    plt.title('PSNR')
    plt.plot(x, bilinear_psnr_sum, color='c', linestyle='--', marker='', label='BILINEAR')
    plt.plot(x, bicubic_psnr_sum, color='k', linestyle='--', marker='', label='BICUBIC')
    plt.plot(x, srcnn_psnr_sum, color='g', linestyle=':', marker='', label='SRCNN')
    # plt.plot(x, edsr_psnr_sum, color='c', linestyle=':', marker='', label='EDSR')
    plt.plot(x, srresnet_psnr_sum, color='y', linestyle='--', label='SRResNet')
    plt.plot(x, srgan_psnr_sum, color='r', linestyle='-.', marker='', label='SRGAN')
    plt.plot(x, asrresnet_psnr_sum, color='m', linestyle='--', label='ASRResNet')
    plt.plot(x, asrgan_psnr_sum, color='b', linestyle='-', marker='', label='ASRGAN')
    plt.xlabel('EPOCH')
    plt.ylabel('PSNR(dB)')
    plt.legend(loc='center right')

    plt.figure(2)
    # plt.subplot(1, 2, 2)
    plt.title('SSIM')
    plt.plot(x, bilinear_ssim_sum, color='c', linestyle='--', marker='', label='BILINEAR')
    plt.plot(x, bicubic_ssim_sum, color='k', linestyle='--', marker='', label='BICUBIC')
    plt.plot(x, srcnn_ssim_sum, color='g', linestyle=':',  marker='', label='SRCNN')
    # plt.plot(x, edsr_ssim_sum, color='c', linestyle=':',  marker='', label='EDSR')
    plt.plot(x, srresnet_ssim_sum, color='y', linestyle='--', label='SRResNet')
    plt.plot(x, srgan_ssim_sum, color='r', linestyle='-.',  marker='', label='SRGAN')
    plt.plot(x, asrresnet_ssim_sum, color='m', linestyle='--', label='ASRResNet')
    plt.plot(x, asrgan_ssim_sum, color='b', linestyle='-',  marker='', label='ASRGAN')
    plt.xlabel('EPOCH')
    plt.ylabel('SSIM')
    # plt.ylim(ymin=0.3, ymax=0.95)
    plt.legend(loc='center right')

    plt.show()
