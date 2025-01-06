# -*- coding: utf-8 -*
"""
MQA(Multi-Query Attention)的实现
"""

import torch
from torch import nn
from torch import Tensor
from math import sqrt

class GroupQueryAttention(nn.Module):
    """
    GQA的实现类
    """
    def __init__(self, hidden_size: int, num_heads: int, group_num: int) -> None:
        """
        初始化函数
        :param hidden_size: 输入特征的维度
        :param num_heads: 分头的数量
        :param group_num: 分组的数量
        """
        super(GroupQueryAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_size // num_heads
        self.group_num = group_num

        # 初始化Q、K、V的投影矩阵
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, self.group_num * self.hidden_dim) ### GQA的特殊做法
        self.v_linear = nn.Linear(hidden_size, self.group_num * self.hidden_dim) ### GQA的特殊做法

        # 初始化一个输出线性层
        self.o_linear = nn.Linear(hidden_size, hidden_size)

        # 定义一个常熟
        self._norm_fact_ = 1 / sqrt(self.hidden_dim)


    def split_heads(self, x: Tensor, group_num=None) -> Tensor:
        """
        分头操作
        :param x:   输入的特征
        :param group_num: 需要分组的数量
        :return:    返回分完组的新特征
        """
        batch_size, seq_len = x.size()[:2]

        if group_num is None:
            return x.view(batch_size, -1, self.num_heads, self.hidden_dim).transpose(1, 2)
        else:
            x = x.view(batch_size, -1, group_num, self.hidden_dim).transpose(1, 2)
            x = x[:, :, None, :, :].expand(batch_size, group_num, self.num_heads // group_num, seq_len, self.hidden_dim)
            x = x.reshape(batch_size, self.num_heads // group_num * group_num, seq_len, self.hidden_dim)
            return x

    # 创建掩码
    def create_mask(self, size: int) -> Tensor:
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


    def forward(self, x: Tensor, attention_mask=None) -> Tensor:
        """
        前向函数
        :param x:  输入的特征
        :param attention_mask: 掩码
        :return: 返回结果
        """
        batch_size = x.shape[0]

        # Q、K、V进行投影
        query = self.q_linear(x)
        key = self.k_linear(x)
        value = self.v_linear(x)

        # 进行分头操作【GQA的特殊操作在这里】
        query = self.split_heads(query)
        key = self.split_heads(key, self.group_num)
        value = self.split_heads(value, self.group_num)

        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self._norm_fact_

        # 看看是否需要进行掩码
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))

        # 对注意力分数进行归一化softmax()
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # 最后乘以value，获得输出
        output = torch.matmul(attention_weights, value)

        # 由于前面的分头操作，需要变形回来
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.hidden_dim)

        # 最后再经过一层全连接层
        output = self.o_linear(output)

        return output


if __name__ == '__main__':
    # Testing GroupQueryAttention
    input_tensor = torch.rand(2, 5, 256)  # batch size = 2, sequence length = 5, embedding dim = 256
    gqa = GroupQueryAttention(hidden_size=256, num_heads=8, group_num=4)

    # 创建一个掩码矩阵
    mask = gqa.create_mask(5)

    output_tensor = gqa(input_tensor, mask)
    print("GroupQueryAttention Output Shape:", output_tensor.shape)  # Expected shape: (2, 5, 256)
