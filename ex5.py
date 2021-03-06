import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


# 欧氏距离
def euclidean(BGR_mean, tmp):
    t = 0
    for i, x in enumerate(BGR_mean):
        t += np.power(tmp[i] - x, 2)
    return np.sqrt(t)


# 曼哈顿距离
def manhattan(BGR_mean, tmp):
    t = 0
    for i, x in enumerate(BGR_mean):
        t += np.abs(tmp[i] - x)
    return t


# 计算距离函数
def distance(func_type: str, img: np.ndarray, BGR_mean: list, value: np.ndarray) -> np.ndarray:
    for i, item in enumerate(img):
        for j, _ in enumerate(item):
            if func_type == "euclidean":
                value[i, j] = euclidean(BGR_mean, img[i, j])
            elif func_type == "manhattan":
                value[i, j] = manhattan(BGR_mean, img[i, j])
    return value


# 通用方法
def general_method(func_type: str, img_to_find: np.ndarray, img_source: np.ndarray, fill_cnt: int = 0) -> np.ndarray:
    BGR = [i for i in cv.split(img_source)]
    BGR_mean = [np.mean(i) for i in BGR]
    value = distance(func_type, img_source, BGR_mean, np.zeros_like(BGR[0]))
    # 计算用于比较的图片，得到比较的avg和std
    avg = np.sum(value) / (img_source.shape[0] * img_source.shape[1])
    std = np.std(value)

    tmp = distance(func_type, img_to_find, BGR_mean, np.zeros(
        (img_to_find.shape[0], img_to_find.shape[1])))
    result = deepcopy(img_to_find)
    # 遍历图片，当符合颜色空间的条件是，则保留，反之置零丢弃
    for i, item in enumerate(tmp):
        for j, _ in enumerate(item):
            if func_type == "euclidean":
                if(np.power(tmp[i, j] - avg, 2) <= std):
                    tmp[i, j] = 255
                else:
                    tmp[i, j] = 0
            elif func_type == "manhattan":
                if(np.abs(tmp[i, j] - avg) <= std):
                    tmp[i, j] = 255
                else:
                    tmp[i, j] = 0
            if tmp[i, j] == 0:
                result[i, j] = np.zeros_like(result[i, j])

    new_result = deepcopy(result)
    # 为避免切割图像部分图像缺失，故采取遍历方法，对周围像素值不为零的点进行还原
    while fill_cnt:
        result = deepcopy(new_result)
        for i in range(1, len(result)-1):
            for j in range(1, len(result[0])-1):
                if result[i-1, j].all() != 0 or result[i, j-1].all() != 0 or result[i+1, j].all() != 0 or result[i, j+1].all() != 0:
                    new_result[i, j] = img_to_find[i, j]
        fill_cnt -= 1
    return new_result


# 使用欧氏距离方法
def euclidean_method(img_to_find: np.ndarray, img_source: np.ndarray, fill_cnt: int = 0) -> np.ndarray:
    return general_method("euclidean", img_to_find, img_source, fill_cnt)


# 使用曼哈顿距离方法
def manhattan_method(img_to_find: np.ndarray, img_source: np.ndarray, fill_cnt: int = 0) -> np.ndarray:
    return general_method("manhattan", img_to_find, img_source, fill_cnt)


# 绘图函数
def draw(image: np.ndarray, strawberry: np.ndarray, fill_cnt: int = 2):
    imgs = [cv.cvtColor(euclidean_method(image, strawberry), cv.COLOR_BGR2RGB), cv.cvtColor(euclidean_method(image, strawberry, fill_cnt), cv.COLOR_BGR2RGB), cv.cvtColor(
        manhattan_method(image, strawberry), cv.COLOR_BGR2RGB), cv.cvtColor(manhattan_method(image, strawberry, fill_cnt), cv.COLOR_BGR2RGB)]
    titles = ['euclidean', 'euclidean with area fill',
              'manhattan', 'manhattan with area fill']
    cnt = len(imgs)
    plt.figure(1, figsize=(10, 5))
    for i in range(cnt):
        plt.subplot(int(f"1{cnt}{i+1}")).set_title(titles[i])
        plt.subplot(int(f"1{cnt}{i+1}")
                    ).imshow(imgs[i])
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)
    plt.show()


if __name__ == '__main__':
    image = cv.imread('./source/ex05.jpg')
    strawberry = cv.imread('./source/strawberry.png')
    # strawberry = cv.imread('./source/little_strawberry.png')

    draw(image, strawberry, 3)
