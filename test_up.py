
import os
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor, Resize


if __name__ == '__main__':

    CROP_SIZE = 512
    UPSCALE_FACTOR = 4
    HR_PATH = f'E:/code/dataset/Set5/set5_HR_{UPSCALE_FACTOR}/'
    LR_PATH = f'E:/code/dataset/Set5/set5_LR_{UPSCALE_FACTOR}_python/'

    lr_scale = Resize(CROP_SIZE // UPSCALE_FACTOR, interpolation=Image.BICUBIC)

    for filename in os.listdir(HR_PATH):
        print(filename)
        hr_image = Image.open(HR_PATH + filename)

        lr_image = lr_scale(hr_image)
        lr_image.save(LR_PATH + 'python_' + filename)

    print('Finish')