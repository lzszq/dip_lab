import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


# 图像相加
def add(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    img1 = img1.astype(int)
    img2 = img2.astype(int)
    result = deepcopy(img1)
    for i, item in enumerate(img1):
        for j, _ in enumerate(item):
            result[i, j] = np.add(img1[i, j], img2[i, j])
            try:
                for k, _ in enumerate(result[i, j]):
                    if result[i, j, k] > 255:
                        result[i, j, k] = 255
            except:
                if result[i, j] > 255:
                    result[i, j] = 255
    return result.astype(np.uint8)


# 图像相减
def subtract(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    img1 = img1.astype(int)
    img2 = img2.astype(int)
    result = deepcopy(img1)
    for i, item in enumerate(img1):
        for j, _ in enumerate(item):
            result[i, j] = np.subtract(img1[i, j], img2[i, j])
            try:
                for k, _ in enumerate(result[i, j]):
                    if result[i, j, k] < 0:
                        result[i, j, k] = 0
            except:
                if result[i, j] < 0:
                    result[i, j] = 0
    return result.astype(np.uint8)


def draw(img1: np.ndarray, img2: np.ndarray, img: np.ndarray, mode: str = "gray"):
    imgs = [img1, img2, img]
    titles = ["img1", "img2", "img"]
    plt.figure(1, figsize=(10, 5))
    for i in range(1, len(imgs)+1):
        plt.subplot(1, 3, i).set_title(titles[i-1])
        if mode == 'gray':
            plt.subplot(1, 3, i
                        ).imshow(imgs[i-1], cmap=mode)
        else:
            plt.subplot(1, 3, i
                        ).imshow(imgs[i-1])
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)
    plt.show()


if __name__ == '__main__':
    img1 = cv.imread('./demo.jpg', cv.IMREAD_GRAYSCALE)
    img2 = cv.imread('./demo1.jpg', cv.IMREAD_GRAYSCALE)
    draw(img1, img2, add(img1, img2), "gray")
    draw(img1, img2, subtract(img1, img2), "gray")

    img1 = cv.cvtColor(cv.imread('./demo.jpg'), cv.COLOR_BGR2RGB)
    img2 = cv.cvtColor(cv.imread('./demo1.jpg'), cv.COLOR_BGR2RGB)
    draw(img1, img2, add(img1, img2), "rgb")
    draw(img1, img2, subtract(img1, img2), "rgb")
