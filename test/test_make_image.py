
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

xmin = 0
xmax = 256
ymin = 0
ymax = 256
box = (xmin, ymin, xmax, ymax)

image_name = '7'

# img_path = f'../data/test_x2/target/{image_name}.png'
img_path = f'C:/Users/lai/Desktop/{image_name}.png'
img = Image.open(img_path).convert("RGB")

img_crop = img.crop(box)
img_crop.save(f'compared/data/crop_{image_name}.png')

draw = ImageDraw.Draw(img)
draw.rectangle([xmin, ymin, xmax, ymax], outline='yellow', width=1)
img.save(f'results/draw_{image_name}.png')

f, ax = plt.subplots(1, 2)
plt.imshow(img)
ax[0].imshow(img)
ax[1].imshow(img_crop)
plt.show()

