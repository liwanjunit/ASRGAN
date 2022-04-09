import cv2 as cv
import numpy as np


def custom_blur_demo(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    dst = cv.filter2D(image, -1, kernel=kernel)
    cv.imshow("custom_blur_demo", dst)


PATH = 'C:/code/ASRGAN/ASRGAN-master/test_image/compared/data_13985_compared.png'

src = cv.imread(PATH)
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
custom_blur_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()
