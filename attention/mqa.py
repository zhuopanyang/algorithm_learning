# -*- coding: utf-8 -*
"""
MQA(Multi-Query Attention)的实现
"""

import torch
from torch import nn
from torch import Tensor
from math import sqrt


class MultiQueryAttention(nn.Module):
    """
    MQA的实现类
    """
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        """
        初始化函数
        :param hidden_size: 输入的特征维度
        :param num_heads:   分头的头数
        """
        super(MultiQueryAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_size // num_heads

        # 初始化Q、K、V的投影矩阵
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, self.hidden_dim) ### MQA的关键操作
        self.v_linear = nn.Linear(hidden_size, self.hidden_dim) ### MQA的关键操作

        # 初始化一个输出线性层
        self.o_linear = nn.Linear(hidden_size, hidden_size)

        # 定义一个常熟
        self._norm_fact_ = 1 / sqrt(self.hidden_dim)


    def split_head(self, x: Tensor, head_num=None) -> Tensor:
        """
        分头操作
        :param x:   输入的特征
        :param head_num:    需要分头的数量
        :return:    返回分头后的新特征
        """
        batch_size = x.shape[0]

        if head_num is None:
            return x.view(batch_size, -1, self.num_heads, self.hidden_dim).transpose(1, 2)
        else:
            return x.view(batch_size, -1, head_num, self.hidden_dim).transpose(1, 2)


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


    def forward(self, x, attention_mask=None) -> Tensor:
        """
        模型的前向函数
        :param x:   输入的特征
        :param attention_mask:  输入的掩码矩阵
        :return:
        """
        batch_size = x.shape[0]

        # Q、K、V进行投影
        query = self.q_linear(x)
        key = self.k_linear(x)
        value = self.v_linear(x)

        # 进行分头操作
        query = self.split_head(query)
        key = self.split_head(key, 1)
        value = self.split_head(value, 1)

        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self._norm_fact_

        # 看是否需要进行掩码
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float("-inf"))

        # 对注意力分数进行归一化softmax()
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # 最后乘以value，获得输出
        output = torch.matmul(attention_weights, value)

        # 由于前面分头，需要重新变形回来
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.hidden_dim)

        # 最后再经过一层全连接层
        output = self.o_linear(output)

        return output
