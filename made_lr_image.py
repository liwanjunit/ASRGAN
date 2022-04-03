
import os
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor, Resize


if __name__ == '__main__':

    CROP_SIZE = 256
    UPSCALE_FACTOR = 4
    TARGET_PATH = 'data/test/target/'
    DATA_PATH = 'data/test/data/'

    lr_scale = Resize(CROP_SIZE // UPSCALE_FACTOR, interpolation=Image.BICUBIC)

    for filename in os.listdir(r'data/test/target'):
        # print(filename)
        hr_image = Image.open(TARGET_PATH + filename)

        lr_image = lr_scale(hr_image)
        lr_image.save(DATA_PATH + filename)

    print('Finish')










