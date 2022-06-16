import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from copy import deepcopy


def get_add(x): return int(x[0]/2)
def get_sub(x): return int(x[0]/2)+1

# 高斯滤波器
def gaussian_filter(img: np.ndarray, D0: int) -> np.ndarray:
    # 计算D
    def cal_D(u, v): return np.sqrt((u[0]-v[0])**2+(u[1]-v[1])**2)
    F = np.fft.fftshift(np.fft.fft2(img))  # 中心变换计算DFT
    H = np.zeros(img.shape)
    center_point = tuple(map(lambda x: int((x-1)/2), img.shape))
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            D = cal_D(center_point, (i, j))
            H[i, j] = np.exp(-(D**2)/(2*(D0**2)))  # 滤波

    result = np.abs(np.fft.ifft2(np.fft.ifftshift(F*H)))  # 计算反DFT，并获取实部
    return result

# 自定义滤波器
def custom_filter(func, img: np.ndarray, kernel_size: tuple = (3, 3)) -> np.ndarray:
    '''
    params:
        func 自定义滤波函数
        img 进行滤波的图片
        kernel_size 卷积核的形状
    '''
    result = deepcopy(img)
    result = result.astype(np.float64)
    img = img.astype(np.float64)
    add = get_add(kernel_size)
    sub = get_sub(kernel_size)
    # 对图片遍历做卷积运算
    for i in range(sub, len(img)+1-sub):
        for j in range(sub, len(img[0])+1-sub):
            result[i, j] = func(img[i-sub:i+add, j-sub:j+add])
    return result.astype(np.uint8)


# laplace算子
def laplace(img: np.ndarray) -> np.ndarray:
    def func(item): return np.abs(np.sum(
        item * np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])))
    return custom_filter(func, img)


# 扩展laplace算子
def laplace_(img: np.ndarray) -> np.ndarray:
    def func(item): return np.abs(np.sum(
        item * np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])))
    return custom_filter(func, img)


# 绘图函数
def draw(img: np.ndarray):
    l1 = laplace(img)
    l2 = laplace_(img)
    imgs = [img, l1, l2]
    titles = ["origin", "laplace", "laplace_"]
    plt.figure(1, figsize=(10, 5))
    for i in range(1, len(imgs)+1):
        plt.subplot(1, 3, i).set_title(titles[i-1])
        plt.subplot(1, 3, i).imshow(imgs[i-1], cmap='gray')
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)
    plt.show()


if __name__ == '__main__':
    img = cv.imread('./source/demo.jpg', cv.IMREAD_GRAYSCALE)
    draw(img)
    draw(gaussian_filter(img, 50))