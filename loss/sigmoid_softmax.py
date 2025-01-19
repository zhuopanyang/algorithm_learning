# -*- coding: utf-8 -*
"""
Sigmoid 和 Softmax 的实现
"""

import numpy as np


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # 减去最大值提高数值稳定性
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def log_loss(y_true, y_pred):
    # 避免对数运算时的数值问题
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    # 计算交叉熵损失
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss
