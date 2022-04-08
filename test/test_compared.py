

import pandas as pd

import matplotlib.pyplot as plt


if __name__ == '__main__':

    srcnn_path = 'C:/code/train_results/new_model/srcnn_test_4.csv'
    srgan_path = 'C:/code/train_results/new_model/srgan_x4/srgan_test_4.csv'
    tsrgan_path = 'C:/code/train_results/new_model/tsrgan_x4/tsrgan_test_4.csv'

    srcnn_data = pd.read_csv(srcnn_path)
    srgan_data = pd.read_csv(srgan_path)
    tsrgan_data = pd.read_csv(tsrgan_path)

    srcnn_psnr_sum = []
    srcnn_ssim_sum = []

    srgan_psnr_sum = []
    srgan_ssim_sum = []

    tsrgan_psnr_sum = []
    tsrgan_ssim_sum = []

    for i in range(len(srgan_data.head(50)['PSNR'])):

        srcnn_psnr_sum.append(srcnn_data['PSNR'][i])
        srcnn_ssim_sum.append(srcnn_data['SSIM'][i])

        srgan_psnr_sum.append(srgan_data['PSNR'][i])
        srgan_ssim_sum.append(srgan_data['SSIM'][i])

        tsrgan_psnr_sum.append(tsrgan_data['PSNR'][i])
        tsrgan_ssim_sum.append(tsrgan_data['SSIM'][i])

    x = range(1, len(srgan_data.head(50)['PSNR'])+1)

    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.plot(x, srcnn_psnr_sum, color="g", linestyle="-.", label="scnan")
    plt.plot(x, srgan_psnr_sum, color="r", linestyle="--", label="srgan")
    plt.plot(x, tsrgan_psnr_sum, color="b", label="tsrgan")
    plt.xlabel("PSNR")
    plt.legend(loc='center right')


    plt.subplot(1, 2, 2)
    plt.plot(x, srcnn_ssim_sum, color="g", linestyle="-.", label="srcnn")
    plt.plot(x, srgan_ssim_sum, color="r", linestyle="--", label="srgan")
    plt.plot(x, tsrgan_ssim_sum, color="b", label="tsrgan")
    plt.xlabel("SSIM")
    plt.legend(loc='center right')

    plt.show()