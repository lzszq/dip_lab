import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


# 二值化函数
def binarization(img: np.ndarray, threshold: int) -> np.ndarray:
    new_img = deepcopy(img)
    for i, item in enumerate(img):
        for j, _ in enumerate(item):
            if img[i, j] >= threshold:
                new_img[i, j] = 255
            else:
                new_img[i, j] = 0
    return new_img


def draw(img: np.ndarray):
    imgs = []
    titles = []
    for threshold in range(50, 200, 10):
        imgs.append(binarization(img, threshold))
        titles.append(threshold)
    plt.figure(1, figsize=(10, 5))
    for i in range(1, len(imgs)+1):
        plt.subplot(3, 5, i).set_title("threshold="+str(titles[i-1]))
        plt.subplot(3, 5, i
                    ).imshow(imgs[i-1], cmap='gray', vmin=0, vmax=255)
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)
    plt.show()


if __name__ == '__main__':
    img = cv.imread('./demo.jpg', cv.IMREAD_GRAYSCALE)
    draw(img)
