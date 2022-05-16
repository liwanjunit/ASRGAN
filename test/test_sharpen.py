import cv2 as cv
import numpy as np


def custom_blur_demo(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    # kernel = np.array([[-1, -1, -1, -1, -1],
    #                    [-1,  2,  2,  2, -1],
    #                    [-1,  2,  8,  2, -1],
    #                    [-1,  2,  2,  2, -1],
    #                    [-1, -1, -1, -1, -1]], np.float32)  # 锐化
    dst = cv.filter2D(image, -1, kernel=kernel)
    cv.imshow("sharpen", dst)


PATH = 'C:/code/ASRGAN/ASRGAN-master/test/compared/data_5345_compared.png'

src = cv.imread(PATH)
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
custom_blur_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()
