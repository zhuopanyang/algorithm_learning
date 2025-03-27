# -*- coding: utf-8 -*
"""
ROPE的实现（有错误的地方，待修改）
"""

from typing import Tuple
import torch


# 生成旋转矩阵
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    i = torch.arange(0, dim // 2).float()  # 获取i，注意这里只需要获取dim//2的量，因为后续两两组合形成复数，维度砍半了
    freqs = 1.0 / (theta ** (2 * i / dim))  # θ_i=10000^(-2i/d) = 1 / (10000^(2i/d))

    m = torch.arange(seq_len, device=freqs.device)  # 生成 token 序列索引 m = [0, 1, ..., seq_len-1]

    freqs = torch.outer(m, freqs).float()  # 计算位置m与频率θ_i的外积 m @ θ_i

    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 这里生成的是复数向量
    return freqs_cis


# 旋转位置编码计算
def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)  # 拆分两两组合，xq_.shape =  [batch_size, seq_len, dim // 2, 2]
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

    xq_ = torch.view_as_complex(xq_)  # 转为复数，xq_.shape = [batch_size, seq_len, dim // 2]
    xk_ = torch.view_as_complex(xk_)

    freqs_cis = freqs_cis[:xq.size(1)]  # 截取与当前序列长度匹配的部分，[max_seq_len * 2, dim // 2] --> [seq_len, dim // 2]

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)  # [batch_size, seq_len, dim // 2] * [seq_len, dim // 2] ---计算---> [batch_size, seq_len, dim // 2]
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)  # ---real---> [batch_size, seq_len, dim // 2, 2] ---flatten---> [batch_size, seq_len, dim]
    return xq_out.type_as(xq), xk_out.type_as(xk)


if __name__ == '__main__':
    dim = 1024
    max_seq_len = 4096
    freqs_cis = precompute_freqs_cis(dim, max_seq_len * 2)  # 这里实际上是做了冗余操作，将旋转矩阵的长度扩充到了2倍的序列长度
    seq_len = 566
    xq = torch.randn(1, seq_len, dim)
    xk = torch.randn(1, seq_len, dim)
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
