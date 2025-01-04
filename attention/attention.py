# -*- coding: utf-8 -*
"""
self-attention 和 multi-head attention的实现
"""

from math import sqrt
from torch import Tensor

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    input : batch_size * seq_len * input_dim
    q : batch_size * input_dim * embed_dim
    k : batch_size * input_dim * embed_dim
    v : batch_size * input_dim * embed_dim
    """

    def __init__(self, input_dim: int, embed_dim: int) -> None:
        """
        self-attention的初始化函数
        :param input_dim:   输入的特征维度
        :param embed_dim: q, k, v 的特征维度
        """
        super(SelfAttention, self).__init__()
        self.q = nn.Linear(input_dim, embed_dim)
        self.k = nn.Linear(input_dim, embed_dim)
        self.v = nn.Linear(input_dim, embed_dim)
        self._norm_fact_ = 1 / sqrt(embed_dim)

    def forward(self, x: Tensor, mask=None) -> Tensor:
        """
        self-attention的前向函数
        :param x:   输入的数据特征
        :param mask: 掩码矩阵
        :return:    返回经过self-attention的计算结果
        """
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        attention_scores = torch.matmul(Q, K.permute(0, 2, 1)) * self._norm_fact_  # Q * K.T()

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0,  float("-inf"))

        attention_weights = torch.softmax(attention_scores, dim=-1)  # 获得 softmax 的输出
        output = torch.matmul(attention_weights, V)  # 获得和 V 的乘积

        return output


class MultiHeadAttention(nn.Module):
    """
    input : batch_size * seq_len * input_dim
    q : batch_size * input_dim * embed_dim
    k : batch_size * input_dim * embed_dim
    v : batch_size * input_dim * embed_dim
    """

    def __init__(self, input_dim: int, embed_dim: int, nums_head: int) -> None:
        """
        multi-attention的初始化函数
        :param input_dim:   输入的特征维度
        :param embed_dim: q, k 和 v 的特征维度
        :param nums_head:   头的数量
        """
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % nums_head == 0

        self.embed_dim = embed_dim
        self.nums_head = nums_head
        self.head_dim = embed_dim // nums_head
        self._norm_fact_ = 1 / sqrt(embed_dim)

        self.q = nn.Linear(input_dim, embed_dim)
        self.k = nn.Linear(input_dim, embed_dim)
        self.v = nn.Linear(input_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor, mask=None) -> Tensor:
        """
        multi-attention的前向函数
        :param x:   输入的数据特征
        :return:    返回multi-attention的计算结果
        """
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        # Split into multiple heads and transpose for computation
        Q = Q.view(x.shape[0], x.shape[1], self.nums_head, self.head_dim).transpose(1, 2)
        K = K.view(x.shape[0], x.shape[1], self.nums_head, self.head_dim).transpose(1, 2)
        V = V.view(x.shape[0], x.shape[1], self.nums_head, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) * self._norm_fact_  # Q * K.T()

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax(attention_scores, dim=-1)    # batch_size * seq_len * seq_len

        output = torch.matmul(attention_weights, V).transpose(1, 2)
        output = output.contiguous().view(x.shape[0], x.shape[1], -1)   # Q * K.T() * V: batch_size * seq_len * dim_v

        return self.fc_out(output)


# 创建掩码
def create_mask(size: int) -> Tensor:
    """
    创建一个掩码矩阵
    :param size:    序列的长度
    :return:    返回一个上三角，或者下三角的掩码矩阵
    """
    # 创建一个上三角矩阵，矩阵的下三角部分设为零
    # mask = torch.triu(torch.ones(size, size), diagonal=1).type(torch.bool)

    # 创建一个下三角矩阵，矩阵的上三角部分设为零
    mask = torch.tril(torch.ones(size, size), diagonal=0).type(torch.bool)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)


if __name__ == '__main__':
    # 创建一个掩码
    mask = create_mask(5)

    # Testing Self-Attention
    input_tensor = torch.rand(2, 5, 256)  # batch size = 2, sequence length = 5, embedding dim = 256
    self_attention = SelfAttention(input_dim=256, embed_dim=256)
    output_tensor = self_attention(input_tensor, mask)
    print("Self-Attention Output Shape:", output_tensor.shape)  # Expected shape: (2, 5, 256)

    # Testing Multi-Head Attention
    multi_head_attention = MultiHeadAttention(input_dim=256, embed_dim=256, nums_head=8)
    output_tensor = multi_head_attention(input_tensor)
    print("Multi-Head Attention Output Shape:", output_tensor.shape)  # Expected shape: (2, 5, 256)
