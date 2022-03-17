
from PIL import Image
from torchvision.transforms import Resize, RandomCrop, ToPILImage

crop = RandomCrop((320, 480))
resize = Resize((160, 240), interpolation=Image.BICUBIC)

for i in range(4):
    path = f'data/{i+1}.png'
    img = Image.open(path)

    img = crop(img)
    img.save(f'test_image/target/{i+1}.png')

    img = resize(img)
    img.save(f'test_image/data/{i+1}.png')







