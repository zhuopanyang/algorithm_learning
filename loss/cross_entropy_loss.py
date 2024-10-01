# -*- coding: utf-8 -*
import torch
from torch import Tensor


def cross_entropy_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """
    cross_entropy_loss的具体实现
    :param logits: 输入的logits分数
    :param labels: 输入的labels
    :return: 返回loss的大小
    """
    # 对logits分数进行softmax操作
    probs = torch.softmax(logits, dim=1)

    # 根据labels选取出相应的probs分数
    true_probs = torch.gather(probs, dim=1, index=labels.unsqueeze(1))

    # 将其取对数
    log_probs = torch.log(true_probs)

    # 求和取平均
    loss = -torch.mean(log_probs)

    return loss


if __name__ == '__main__':
    # cross_entrop_loss的测试代码
    logits = torch.rand(64, 10)
    labels = torch.randint(0, 10, (64,))

    loss = cross_entropy_loss(logits, labels)
    print("Cross_entropy_loss：", loss)
