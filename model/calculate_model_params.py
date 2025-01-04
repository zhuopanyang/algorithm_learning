# -*- coding: utf-8 -*
"""
计算Transformer、CNN的参数量

假如是一个网络模型model，则可以如下代码，来获得模型的参数量
total_params = sum(p.numel() for p in model.parameters())

"""

def calculate_transformer_params(d_model, n_heads, d_ff, n_layers) -> int:
    """
    计算Transformer的参数量
    :param d_model: 模型的维度，即词向量的维度。
    :param n_heads: 多头自注意力机制中的头数。
    :param d_ff: 前馈网络中的隐藏层维度。
    :param n_layers: Transformer模型中的层数（编码器和解码器的层数之和）
    :return:    返回该Transformer的参数量
    """
    # 多头自注意力层参数量 = QKV的三个权重矩阵 + 输出层的权重矩阵
    attention_params = n_layers * (3 * d_model * d_model)  # Q, K, V的三个投影权重矩阵
    attention_params += n_layers * (d_model * d_model)  # 输出层的权重矩阵参数

    # 前馈网络参数量 = 两个线性层
    ff_params = n_layers * (d_model * d_ff + d_ff * d_model)  # FFN

    # 层归一化参数量 = 两个去参数（一个缩放参数gamma、一个偏移参数beta）
    layer_norm_params = n_layers * (2 * d_model)  # 每层两个参数

    # 总参数量
    total_params = attention_params + ff_params + layer_norm_params
    return total_params


def calculate_cnn_params(input_channels, output_channels, kernel_size, stride, padding, n_layers) -> int:
    """
    计算CNN的参数量
    :param input_channels: 输入通道数
    :param output_channels: 输出通道数
    :param kernel_size: 卷积核的大小
    :param stride: 卷积步长
    :param padding: 填充大小
    :param n_layers: 卷积层的数量
    :return:    返回CNN的参数量
    """
    # 每层卷积参数量 = 每个卷积核的参数量为 kernel_size * kernel_size，乘以input_channels * output_channels
    conv_params_per_layer = (kernel_size * kernel_size) * input_channels * output_channels
    # 总卷积参数量
    conv_params = conv_params_per_layer * n_layers

    # 假设每个卷积层后都有一个偏置项（每个卷积层的每个输出通道都有一个偏置项）
    bias_params = output_channels * n_layers

    # 假设使用全连接层，输出特征数为10
    fc_input_features = output_channels * (kernel_size // stride) ** 2  # 卷积神经网络输出的特征图大小，作为线性层的输入
    fc_params = fc_input_features * 10 + 10 # 全连接层的参数量 = 输入 * 输出 + 偏置项

    # 总参数量
    total_params = conv_params + bias_params + fc_params
    return total_params


if __name__ == '__main__':
    # 计算Transformer的参数量
    d_model = 512
    n_heads = 8
    d_ff = 2048
    n_layers = 6

    transformer_params = calculate_transformer_params(d_model, n_heads, d_ff, n_layers)
    print(f"Transformer参数量: {transformer_params}")

    # 计算CNN的参数量
    input_channels = 3
    output_channels = [64, 128, 256]  # 假设有三个卷积层
    kernel_size = 3
    stride = 1
    padding = 1
    n_layers = len(output_channels)  # 卷积层数

    cnn_params = calculate_cnn_params(input_channels, output_channels[0], kernel_size, stride, padding, n_layers)
    for i in range(1, n_layers):
        cnn_params += calculate_cnn_params(output_channels[i - 1], output_channels[i], kernel_size, stride, padding, 1)

    print(f"CNN参数量: {cnn_params}")
