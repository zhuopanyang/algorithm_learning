# -*- coding: utf-8 -*
"""
    推荐算法中，常见模型的实现及其使用
    FM、Wide & Deep、DeepFM、DCN、NCF
"""
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    FM模型的实现 = 线性项 + 交互项
    因子分解机（Factorization Machine，简称FM）是一种强大的机器学习算法
    核心思想是通过矩阵分解来捕捉特征之间的交互作用
"""
class FM(nn.Module):
    """
    FM模型（因子分解机）
    """
    def __init__(self, num_fields: int, embed_dim: int) -> None:
        super(FM, self).__init__()
        self.num_fields = num_fields
        self.embed_dim = embed_dim

        # 为每个特征创建一个嵌入层
        self.embeddings = nn.Embedding(num_fields, embed_dim)

        # 为每个特征创建一个线性权重
        self.linear_weights = nn.Parameter(torch.randn(num_fields))
        # 也可以设置一个线性层
        # self.linear = nn.Embedding(num_fields, 1)

    def forward(self, x: Tensor) -> Tensor:
        # （1）计算线性部分
        linear_part = torch.sum(self.linear_weights * x, dim=1, keepdim=True)

        # 将输入的特征索引转换为嵌入向量
        embeddings = self.embeddings(x)
        # （2）计算二阶特征交叉部分
        square_of_sum = torch.sum(embeddings, dim=1, keepdim=True).pow(2)
        sum_of_square = torch.sum(embeddings.pow(2), dim=1, keepdim=True)
        cross_part = 0.5 * torch.sub(square_of_sum, sum_of_square)

        # 合并线性部分和二阶特征交叉部分
        total = linear_part + cross_part

        # 应用sigmoid函数得到预测结果
        return torch.sigmoid(total)


def test_fm():
    # 假设参数
    num_fields = 10  # 特征字段的数量
    embed_dim = 5  # 嵌入向量的维度

    # 创建FM模型实例
    model = FM(num_fields, embed_dim)

    # 假设输入数据，每个特征字段的索引
    x = torch.randint(0, num_fields, (32, num_fields))  # 32个样本，每个样本有num_fields个特征

    # 前向传播
    predictions = model(x)
    print(predictions)


"""
    Wide & Deep模型的实现 = Wide部分 + Deep部分（深度神经网络）
    Wide部分是一个广义线性模型，主要用于捕捉特征之间的显式交互。
    Deep部分是一个深度神经网络，用于学习特征之间的隐式交互。
"""
# 定义Wide部分
class LinearPart(nn.Module):
    """
    Wide部分
    """
    def __init__(self, input_dim: int) -> None:
        super(LinearPart, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


# 定义Deep部分
class DeepPart(nn.Module):
    """
    Deep部分
    """
    def __init__(self, input_dim: int, hidden_units: list, dropout: int) -> None:
        super(DeepPart, self).__init__()
        # 定义多层 线性层 + relu() + dropout()
        layers = []
        for i, unit in enumerate(hidden_units):
            if i == 0:
                layers.append(nn.Linear(input_dim, unit))
            else:
                layers.append(nn.Linear(hidden_units[i - 1], unit))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


# 定义Wide & Deep模型
class WideDeepModel(nn.Module):
    """
    Wide & Deep模型
    """
    def __init__(self, input_dim: int, hidden_units: list, dropout: int) -> None:
        super(WideDeepModel, self).__init__()
        # 定义Wide部分
        self.linear_part = LinearPart(input_dim)
        # 定义Deep部分
        self.deep_part = DeepPart(input_dim, hidden_units, dropout)
        self.final_linear = nn.Linear(hidden_units[-1] + 1, 1)  # 合并Wide和Deep部分

    def forward(self, wide_input: Tensor, deep_input: Tensor) -> Tensor:
        wide_output = self.linear_part(wide_input)
        deep_output = self.deep_part(deep_input)
        combined_output = torch.cat((wide_output, deep_output), dim=1)
        return torch.sigmoid(self.final_linear(combined_output))


def test_wide_deep():
    # 假设参数
    input_dim = 13  # 输入特征的维度
    hidden_units = [256, 128, 64]  # Deep部分的隐藏层维度
    dropout = 0.5  # Dropout比率

    # 创建Wide & Deep模型实例
    model = WideDeepModel(input_dim, hidden_units, dropout)

    # 假设输入数据
    wide_input = torch.rand(32, input_dim) # 32个样本
    deep_input = torch.randn(32, input_dim) # 32个样本

    # 前向传播
    predictions = model(wide_input, deep_input)
    print(predictions)


"""
    DeepFM模型的实现 = FM（因子分解机） + DNN（深度神经网络）：同时学习低阶和高阶特征交互
    FM通过引入隐向量来表示特征之间的交互，能够自动学习特征之间的交叉关系。
    DNN负责学习高阶特征交互。DNN由嵌入层（Embedding Layer）和隐藏层（Hidden Layer）组成，通过多层感知机（MLP）来捕捉复杂的特征交互。
"""
# 定义DeepFM模型
class DeepFM(nn.Module):
    """
    DeepFM模型的实现
    """
    def __init__(self, num_features: int, embedding_dim: int, hidden_units: list, output_dim: int, dropout: int) -> None:
        super(DeepFM, self).__init__()
        # FM部分
        self.embedding = nn.Embedding(num_features, embedding_dim)
        self.linear = nn.Embedding(num_features, 1)

        # DNN部分
        # 定义多层 线性层 + relu() + dropout()
        layers = []
        for i, unit in enumerate(hidden_units):
            if i == 0:
                layers.append(nn.Linear(num_features, unit))
            else:
                layers.append(nn.Linear(hidden_units[i - 1], unit))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
        self.mlp = nn.Sequential(*layers)

        # 输出层
        self.output_layer = nn.Linear(output_dim + hidden_units[-1], output_dim)

    def forward(self, wide_input: Tensor, deep_input: Tensor) -> Tensor:
        # FM部分
        fm_embeddings = self.embedding(wide_input)
        fm_linear = self.linear(wide_input).squeeze(2)
        # fm_first_order = torch.sum(fm_linear, dim=1)
        fm_second_order = 0.5 * (torch.sum(fm_embeddings, dim=1) ** 2 - torch.sum(fm_embeddings ** 2, dim=1))
        fm_output = fm_linear + fm_second_order

        # DNN部分
        dnn_output = self.mlp(deep_input)

        # 合并FM和DNN的输出
        combined_output = torch.cat((fm_output, dnn_output), dim=1)
        final_output = self.output_layer(combined_output)
        return final_output


def test_deepfm():
    # 假设参数
    num_features = 10   # 输入特征向量的维度
    embedding_dim = 10 # 嵌入向量的维度
    hidden_units = [256, 128, 64]   # DNN的隐藏层的维度
    output_dim = 10  # 嵌入向量的维度
    dropout = 0.5   #dropout的设置

    # 创建DeepFM模型实例
    model = DeepFM(num_features, embedding_dim, hidden_units, output_dim, dropout)

    # 假设输入数据，每个field的索引
    wide_input = torch.randint(0, 10, size=(32, num_features))  # 32个样本，10个field
    deep_input = torch.rand(32, num_features)  # 32个样本，10个field

    # 前向传播
    predictions = model(wide_input, deep_input)
    print(predictions)


"""
    DCN模型的实现 (Deep & Cross Network) = Cross Network + DNN（深度神经网络）
    Cross Network显式地构建高阶特征交互，并将其与原始特征结合起来，形成新的特征表示
    DNN负责学习高阶特征交互。DNN由嵌入层（Embedding Layer）和隐藏层（Hidden Layer）组成，通过多层感知机（MLP）来捕捉复杂的特征交互。
"""
class CrossNetwork(nn.Module):
    """
    Cross Network的实现
    """
    def __init__(self, input_dim: int, num_layers: int) -> None:
        super(CrossNetwork, self).__init__()
        self.num_layers = num_layers
        self.linears = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_layers)])

    def forward(self, x: Tensor) -> Tensor:
        x_list = [x]

        for i in range(self.num_layers):
            x_ = x_list[-1]
            for j, x_pre in enumerate(x_list[:-1]):
                x_ = x_ + (x_ * x_pre).sum(dim=-1, keepdim=True)
            x_list.append(self.linears[i](x_))

        out = torch.cat(x_list[1:], dim=-1)
        return out


class DeepNetwork(nn.Module):
    """
    DNN网络的实现
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super(DeepNetwork, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class DCN(nn.Module):
    """
    DCN模型的实现
    """
    def __init__(self, input_dim: int, num_layers: int, hidden_dim: int, output_dim: int) -> None:
        super(DCN, self).__init__()
        self.cross_network = CrossNetwork(input_dim, num_layers)
        self.deep_network = DeepNetwork(input_dim, hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        cross_out = self.cross_network(x)
        deep_out = self.deep_network(x)
        return torch.sigmoid(cross_out + deep_out)


def test_dcn():
    # 假设参数
    input_dim = 10  # 输入特征的维度
    num_layers = 3  # 交叉网络的层数
    hidden_dim = 64  # 深度网络的隐藏层维度
    output_dim = 1  # 输出维度

    # 创建DCN模型实例
    model = DCN(input_dim, num_layers, hidden_dim, output_dim)

    # 假设输入数据
    x = torch.randn(32, input_dim)  # 32个样本

    # 前向传播
    predictions = model(x)
    print(predictions)


"""
    NCF模型的实现(Neural Collaborative Filtering)是一种基于神经网络的协同过滤模型
    传统的协同过滤方法，如矩阵分解（Matrix Factorization, MF），通常通过内积来建模用户和项目的潜在特征交互，但可能无法捕捉用户交互数据的复杂结构
    NCF通过引入神经网络，能够学习用户和项目之间的非线性交互关系
"""
class NCF(nn.Module):
    """
    NCF模型的实现
    """
    def __init__(self, num_users: int, num_items: int, embed_size: int, mlp_layers: int) -> None:
        super(NCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_size = embed_size

        # 用户和物品嵌入层
        self.user_embedding = nn.Embedding(num_users, embed_size)
        self.item_embedding = nn.Embedding(num_items, embed_size)

        # 广义矩阵分解（GMF）部分
        self.gmf = nn.Embedding(num_users, embed_size)

        # 多层感知机（MLP）部分
        self.mlp_layers = nn.ModuleList()
        self.mlp_layers.append(nn.Linear(embed_size * 2, mlp_layers[0]))
        for i in range(1, len(mlp_layers)):
            self.mlp_layers.append(nn.Linear(mlp_layers[i - 1], mlp_layers[i]))
        self.mlp_layers.append(nn.Linear(mlp_layers[-1], 1))

    def forward(self, user_indices, item_indices):
        # 用户和物品嵌入
        user_embedding = self.user_embedding(user_indices)
        item_embedding = self.item_embedding(item_indices)

        # GMF
        gmf_vector = user_embedding * item_embedding

        # MLP
        mlp_vector = torch.cat([user_embedding, item_embedding], dim=1)
        for layer in self.mlp_layers[:-1]:
            mlp_vector = F.relu(layer(mlp_vector))
        mlp_vector = self.mlp_layers[-1](mlp_vector)

        # 合并GMF和MLP的输出
        pred = torch.sigmoid(gmf_vector.sum(dim=1) + mlp_vector)

        return pred


def test_ncf():
    # 假设参数
    num_users = 1000
    num_items = 1000
    embed_size = 64
    mlp_layers = [128, 64]

    # 创建NCF模型实例
    model = NCF(num_users, num_items, embed_size, mlp_layers)

    # 假设输入的用户和物品索引
    user_indices = torch.randint(0, num_users, (32,))  # 32个用户
    item_indices = torch.randint(0, num_items, (32,))  # 32个物品

    # 前向传播
    predictions = model(user_indices, item_indices)
    print(predictions)


if __name__ == '__main__':
    test_ncf()
