
from PIL import Image
from torchvision.transforms import Resize, RandomCrop, ToPILImage

crop_size = 64


crop = RandomCrop((320, 480))
lr_scale = Resize(crop_size, interpolation=Image.BICUBIC)

for i in range(358):
    path = f'data/test/target/data_{i+13643}.png'
    hr_img = Image.open(path)

    lr_img = lr_scale(hr_img)
    lr_img.save(f'data/test/data/data_{i+13643}.png')







