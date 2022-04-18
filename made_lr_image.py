
import os
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor, Resize


if __name__ == '__main__':

    CROP_SIZE = 256
    UPSCALE_FACTOR = 2
    TARGET_PATH = f'data/test_x{UPSCALE_FACTOR}/target/'
    LR_PATH = f'data/test_x{UPSCALE_FACTOR}/data/'
    BICUBIC_PATH = f'data/test_x{UPSCALE_FACTOR}/bicubic/'

    lr_scale = Resize(CROP_SIZE // UPSCALE_FACTOR, interpolation=Image.BICUBIC)
    bicubic_scale = Resize(CROP_SIZE, interpolation=Image.BICUBIC)

    for filename in os.listdir(rf'data/test_x{UPSCALE_FACTOR}/target'):
        # print(filename)
        hr_image = Image.open(TARGET_PATH + filename)

        lr_image = lr_scale(hr_image)
        bicubic_image = bicubic_scale(lr_image)
        bicubic_image.save(BICUBIC_PATH + filename)
        lr_image.save(LR_PATH + filename)

    print('Finish')










