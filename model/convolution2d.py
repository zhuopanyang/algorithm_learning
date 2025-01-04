# -*- coding: utf-8 -*
import numpy as np


def convolution2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    二维卷积操作。

    参数:
    image -- 输入图像，一个二维NumPy数组。
    kernel -- 卷积核，一个二维NumPy数组。

    返回:
    卷积后的图像，一个二维NumPy数组。
    """
    # 获取图像和卷积核的维度
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # 计算卷积后的图像尺寸
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    # 创建输出图像数组
    output = np.zeros((output_height, output_width))

    # 进行卷积操作
    for y in range(output_height):
        for x in range(output_width):
            # 提取当前窗口
            window = image[y:y + kernel_height, x:x + kernel_width]
            # 计算卷积和
            output[y, x] = np.sum(window * kernel)

    return output


if __name__ == '__main__':
    # 示例使用
    # 创建一个简单的图像（5x5的数组）
    image = np.array([
        [1, 2, 3, 3, 2],
        [2, 3, 4, 4, 3],
        [3, 4, 5, 5, 4],
        [3, 4, 5, 5, 4],
        [2, 3, 4, 4, 3]
    ])

    # 创建一个简单的卷积核（3x3的数组）
    kernel = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])

    # 执行卷积操作
    result = convolution2d(image, kernel)

    # 打印结果
    print(result)
