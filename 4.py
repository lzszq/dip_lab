import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import sys
sys.setrecursionlimit(100000)


def get_data(addr: str = './demo.jpg') -> np.ndarray:
    return cv.imread(addr, cv.IMREAD_GRAYSCALE)


# canny边缘检测
def canny(img: np.ndarray, D0: int, threshold_low: int, threshold_high: int) -> np.ndarray:
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

    # sobel 算子
    def sobel(img: np.ndarray, direction: str = 'xy') -> (np.ndarray, np.ndarray):
        H1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        H2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        # funcxy = lambda A: np.sqrt(np.power(np.sum(H1*A), 2)+np.power(np.sum(H2*A), 2))
        def funcxy(A): return np.abs(np.sum(H1*A)+np.sum(H2*A))
        def funcx(A): return np.abs(np.sum(H1*A))
        def funcy(A): return np.abs(np.sum(H2*A))
        def func_phase(A): return np.arctan(np.sum(H2*A)/np.sum(H1*A))
        result = deepcopy(img)
        directions = np.zeros([img.shape[0], img.shape[1]])
        if direction == 'x':
            func = funcx
        elif direction == 'y':
            func = funcy
        elif direction == 'xy':
            func = funcxy
        # 循环扫描，计算梯度幅值和梯度方向
        for i in range(1, len(img)-1):
            for j in range(1, len(img[0])-1):
                A = img[i-1:i+2, j-1:j+2]
                result[i, j] = func(A)
                if funcx(A) == 0:
                    directions[i, j] = np.pi / 2
                else:
                    directions[i, j] = func_phase(A)
        return (result.astype(np.uint8), directions)

    # 非极大值抑制
    def NMS(gradients: np.ndarray, directions: np.ndarray) -> np.ndarray:
        nms = deepcopy(gradients[1:-1, 1:-1])
        # 判断方向获取判断参数

        def get_t_weight(theta, weight):
            T = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [
                         1, 0, 1, -1], [0, -1, 1, -1]])
            if theta > np.pi/4:
                t = T[0]
                weight = 1/weight
            elif theta >= 0:
                t = T[1]
            elif theta >= -np.pi/4:
                t = T[2]
                weight *= -1
            else:
                t = T[3]
                weight = -1/weight
            return t, weight

        # 循环扫描
        for i in range(1, gradients.shape[0]-1):
            for j in range(1, gradients.shape[1]-1):
                theta = directions[i, j]
                weight = np.tan(theta)
                t, weight = get_t_weight(theta, weight)
                g = []
                ope = [1, -1]
                for k in range(2):
                    g.append(gradients[i+ope[k]*t[0], j+ope[k]*t[1]])
                    g.append(gradients[i+ope[k]*t[2], j+ope[k]*t[3]])
                if g[0]*weight+g[1]*(1-weight) > gradients[i, j] or g[2]*weight+g[3]*(1-weight) > gradients[i, j]:
                    nms[i-1, j-1] = 0
        return nms

    def double_threshold(nms, threshold_low, threshold_high):
        visited = np.zeros_like(nms)
        output_image = deepcopy(nms)
        W, H = output_image.shape
        # 深度优先遍历，将符合条件的值置为255

        def dfs(output_image, i, j):
            if i >= W or i < 0 or j >= H or j < 0 or visited[i, j] == 1:
                return
            visited[i, j] = 1
            if output_image[i, j] > threshold_low:
                output_image[i, j] = 255
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        dfs(output_image, i+k, j+l)
            else:
                output_image[i, j] = 0
        # 循环扫描
        for w in range(W):
            for h in range(H):
                if visited[w, h] == 1:
                    continue
                if output_image[w, h] >= threshold_high:
                    dfs(output_image, w, h)
                elif output_image[w, h] <= threshold_low:
                    output_image[w, h] = 0
                    visited[w, h] = 1
        # 将未达的点置0
        for w in range(W):
            for h in range(H):
                if visited[w, h] == 0:
                    output_image[w, h] = 0
        return output_image

    gaussian = gaussian_filter(img, D0)
    gradients, directions = sobel(gaussian)
    nms = NMS(gradients, directions)
    result = double_threshold(nms, threshold_low, threshold_high)
    return result.astype(np.uint8)


def show_canny_result(img: np.ndarray, D0: int, threshold_low: int, threshold_high: int):
    imgs = [img, canny(img, D0, threshold_low, threshold_high)]
    titles = ['origin', 'canny']
    cnt = len(imgs)
    plt.figure(1, figsize=(10, 5))
    for i in range(cnt):
        plt.subplot(int(f"1{cnt}{i+1}")).set_title(titles[i])
        plt.subplot(int(f"1{cnt}{i+1}")
                    ).imshow(imgs[i], cmap='gray')
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)
    plt.show()


if __name__ == '__main__':
    img = get_data('./demo.jpg')
    show_canny_result(img, 50, 10, 30)
    show_canny_result(img, 50, 30, 90)
    show_canny_result(img, 50, 50, 150)
