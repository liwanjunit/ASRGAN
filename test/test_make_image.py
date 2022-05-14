

import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


xmin = 140
xmax = 220
ymin = 40
ymax = 120
box = (xmin, ymin, xmax, ymax)

image_name = 'data_62'

img_path = f'../data/test_x2/target/{image_name}.png'
img = Image.open(img_path)

img_crop = img.crop(box)
img_crop.save(f'results/crop_{image_name}.png')

draw = ImageDraw.Draw(img)
draw.rectangle([xmin, ymin, xmax, ymax], outline='yellow', width=1)
img.save(f'results/draw_{image_name}.png')

f, ax = plt.subplots(1, 2)
plt.imshow(img)
ax[0].imshow(img)
ax[1].imshow(img_crop)
plt.show()

