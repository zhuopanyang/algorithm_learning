# -*- coding: utf-8 -*
"""
    推荐算法中，常见模型的实现及其使用
    FM、Wide & Deep、DeepFM、DCN、NCF
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    FM模型的实现
"""
class FM(nn.Module):
    def __init__(self, num_fields, embed_dim):
        super(FM, self).__init__()
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        # 为每个特征创建一个嵌入层
        self.embeddings = nn.Embedding(num_fields, embed_dim)
        # 为每个特征创建一个线性权重
        self.linear_weights = nn.Parameter(torch.randn(num_fields))

    def forward(self, x):
        # 将输入的特征索引转换为嵌入向量
        embeddings = self.embeddings(x)
        # 计算线性部分
        linear_part = torch.sum(self.linear_weights * x, dim=1, keepdim=True)
        # 计算二阶特征交叉部分
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
    Wide & Deep模型的实现
"""
# 定义Wide部分
class LinearPart(nn.Module):
    def __init__(self, input_dim):
        super(LinearPart, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


# 定义Deep部分
class DeepPart(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout):
        super(DeepPart, self).__init__()
        layers = []
        for i, unit in enumerate(hidden_units):
            if i == 0:
                layers.append(nn.Linear(input_dim, unit))
            else:
                layers.append(nn.Linear(hidden_units[i - 1], unit))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(hidden_units[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# 定义Wide & Deep模型
class WideDeepModel(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout):
        super(WideDeepModel, self).__init__()
        self.linear_part = LinearPart(input_dim)
        self.deep_part = DeepPart(input_dim, hidden_units, dropout)
        self.final_linear = nn.Linear(hidden_units[-1] + 1, 1)  # 合并Wide和Deep部分

    def forward(self, x):
        wide_output = self.linear_part(x)
        deep_output = self.deep_part(x[:, len(self.linear_part.linear.weight):])
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
    wide_input = torch.randn(32, input_dim)  # 32个样本

    # 前向传播
    predictions = model(wide_input)
    print(predictions)


"""
    DeepFM模型的实现
"""
class DeepFM(nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_layers):
        super(DeepFM, self).__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.linear = nn.ModuleList([nn.Linear(1, 1) for _ in range(len(field_dims))])
        self.mlp = nn.Sequential()
        input_dim = embed_dim * len(field_dims)
        for i in range(len(mlp_layers)):
            if i == 0:
                self.mlp.add_module('linear%d' % (i + 1), nn.Linear(input_dim, mlp_layers[i]))
            else:
                self.mlp.add_module('linear%d' % (i + 1), nn.Linear(mlp_layers[i - 1], mlp_layers[i]))
            if i != len(mlp_layers) - 1:
                self.mlp.add_module('relu%d' % (i + 1), nn.ReLU())
                self.mlp.add_module('dropout%d' % (i + 1), nn.Dropout(p=0.2))
        self.mlp.add_module('output', nn.Linear(mlp_layers[-1], 1))
        self.logits = nn.Linear(embed_dim * len(field_dims), 1)

    def forward(self, x):
        embeds = self.embedding(x).squeeze(1)
        linear_out = torch.cat([linear(embeds) for linear in self.linear], 1)
        deep_out = F.relu(self.mlp(embeds))
        return torch.sigmoid(self.logits(deep_out).squeeze(1)) + linear_out


def test_deepfm():
    # 假设参数
    field_dims = [10, 8, 5]  # 假设有三个field，维度分别为10, 8, 5
    embed_dim = 4  # 嵌入向量的维度
    mlp_layers = [128, 64]  # MLP层的维度

    # 创建DeepFM模型实例
    model = DeepFM(field_dims, embed_dim, mlp_layers)

    # 假设输入数据，每个field的索引
    x = torch.randint(0, 10, (32, 3))  # 32个样本，3个field

    # 前向传播
    predictions = model(x)
    print(predictions)


"""
    DCN模型的实现
"""
class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(CrossNetwork, self).__init__()
        self.num_layers = num_layers
        self.linears = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_layers)])

    def forward(self, x):
        x_list = [x]

        for i in range(self.num_layers):
            x_ = x_list[-1]
            for j, x_pre in enumerate(x_list[:-1]):
                x_ = x_ + (x_ * x_pre).sum(dim=-1, keepdim=True)
            x_list.append(self.linears[i](x_))

        out = torch.cat(x_list[1:], dim=-1)
        return out


class DeepNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeepNetwork, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class DCN(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dim, output_dim):
        super(DCN, self).__init__()
        self.cross_network = CrossNetwork(input_dim, num_layers)
        self.deep_network = DeepNetwork(input_dim, hidden_dim, output_dim)

    def forward(self, x):
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
    NCF模型的实现
"""
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embed_size, mlp_layers):
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
