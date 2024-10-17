# -*- coding: utf-8 -*
"""
写出一个使用 MSE loss 来训练神经网络模型的训练步骤
"""
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch import Tensor
from torch.utils.data import TensorDataset

# 设置随机种子以保证结果重复性
torch.manual_seed(1)


class SimpleMLP(nn.Module):
    """
    定义一个简单的三层MLP模型
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """
        初始化函数
        :param input_size:  输入的数据特征的维度大小
        :param hidden_size: 中间特征的隐藏维度大小
        :param output_size: 输出的特征维度的大小
        """
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向函数
        :param x:   输入的数据特征
        :return:    返回模型输出的 logits 分数
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


# 超参数定义
input_size = 10
hidden_size = 50
output_size = 1
num_epochs = 100
learning_rate = 0.01
batch_size = 16  # 每个批次的样本数量

# 初始化模型、损失函数和优化器
model = SimpleMLP(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 随机输入和输出数据（假设）
x_train = torch.randn(100, input_size)  # 100个样本，每个样本有input_size个特征
y_train = torch.randn(100, output_size)  # 100个目标值

# 创建 TensorDataset 和 DataLoader
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 训练循环
for epoch in range(num_epochs):
    model.train()  # 将模型设为训练模式

    for inputs, labels in train_loader:
        # 正向传播
        outputs = model(x_train)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新模型参数

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete.")
