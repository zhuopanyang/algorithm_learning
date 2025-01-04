import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """
        计算对比损失
        :param output1: 网络的第一个输出，形状为[N, D]
        :param output2: 网络的第二个输出，形状为[N, D]
        :param label: 标签，1表示正样本对，0表示负样本对，形状为[N]
        :return: 对比损失值
        """
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)))
        return loss_contrastive

# 示例使用
# 假设我们有两个输出和对应的标签
output1 = torch.randn(10, 128)  # 假设有10个样本，每个样本128维
output2 = torch.randn(10, 128)
label = torch.randint(0, 2, (10,))  # 随机生成0或1作为标签

# 创建损失函数实例
criterion = ContrastiveLoss(margin=1.0)

# 计算损失
loss = criterion(output1, output2, label)
print("Contrastive Loss:", loss.item())