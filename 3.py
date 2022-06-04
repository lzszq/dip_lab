import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from copy import deepcopy


def get_add(x): return int(x[0]/2)
def get_sub(x): return int(x[0]/2)+1


def custom_filter(func, img: np.ndarray, kernel_size: tuple = (3, 3)) -> np.ndarray:
    result = deepcopy(img)
    add = get_add(kernel_size)
    sub = get_sub(kernel_size)
    for i in range(sub, len(img)+1-sub):
        for j in range(sub, len(img[0])+1-sub):
            result[i, j] = func(img[i-sub:i+add, j-sub:j+add])
    return result


def laplace(img: np.ndarray) -> np.ndarray:
    def func(item): return np.abs(np.sum(
        item * np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])))
    return custom_filter(func, img)


def draw(img: np.ndarray):
    imgs = [img, laplace(img)]
    titles = ["origin", "laplace"]
    plt.figure(1, figsize=(10, 5))
    for i in range(1, len(imgs)+1):
        plt.subplot(1, 2, i).set_title(titles[i-1])
        plt.subplot(1, 2, i).imshow(imgs[i-1], cmap='gray')
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)
    plt.show()


if __name__ == '__main__':
    img = cv.imread('./demo.jpg', cv.IMREAD_GRAYSCALE)
    draw(img)