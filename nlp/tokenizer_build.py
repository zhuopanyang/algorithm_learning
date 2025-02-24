# -*- coding: utf-8 -*
"""

创建一个Tokenizer的类

"""
import torch

from torch import Tensor


class Tokenizer:
    """
    实现一个Tokenizer类
    """
    def __init__(self, special_tokens=None):
        """
        初始化函数
        :param special_tokens:  Token的特殊标记
        """
        # 设置默认特殊标记，确保包含[UNK]
        default_special = ['[PAD]', '[UNK]']
        if special_tokens is None:
            special_tokens = default_special
        else:
            if '[UNK]' not in special_tokens:
                special_tokens.append('[UNK]')
        self.special_tokens = special_tokens
        self.stoi = {}  # 字符串到ID的映射
        self.itos = {}  # ID到字符串的映射

        # 初始化时添加特殊标记
        self._add_tokens(self.special_tokens)

    def _add_tokens(self, tokens: list):
        """
        将一组标记添加到词汇表中
        :param tokens:  输入的Token标记
        :return:    None
        """
        for token in tokens:
            if token not in self.stoi:
                idx = len(self.stoi)
                self.stoi[token] = idx
                self.itos[idx] = token

    def build_vocab(self, corpus: list):
        """
        从语料库中构建词汇表
        :param corpus:  语料库
        :return:    None
        """
        # 收集所有唯一字符
        chars = set()
        for text in corpus:
            chars.update(text)
        # 按Unicode编码排序字符以确保一致性
        sorted_chars = sorted(chars)
        # 添加字符到词汇表（跳过已存在的）
        self._add_tokens(sorted_chars)

    def encode(self, text: str):
        """
        将文本编码为张量
        :param text: 输入的文本信息
        :return:    返回文本的张量
        """
        unk_id = self.stoi['[UNK]']
        ids = [self.stoi.get(c, unk_id) for c in text]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, tensor: Tensor):
        """
        将张量解码为文本
        :param tensor:
        :return:
        """
        ids = tensor.tolist()
        return ''.join([self.itos.get(i, '[UNK]') for i in ids])


# 示例用法
if __name__ == "__main__":
    # 初始化分词器并指定特殊标记
    tokenizer = Tokenizer(special_tokens=['[PAD]', '[BOS]', '[EOS]'])

    # 训练语料
    corpus = [
        "hello world!",
        "This is a test.",
        "PyTorch tokenizer example."
    ]

    # 构建词汇表
    tokenizer.build_vocab(corpus)

    # 编码测试
    text = "hello PyTorch!"
    encoded = tokenizer.encode(text)
    print(f"Encoded '{text}':\n{encoded}")

    # 解码测试
    decoded = tokenizer.decode(encoded)
    print(f"Decoded back: {decoded}")

    # 显示词汇表信息
    print(f"\nVocabulary size: {len(tokenizer.stoi)}")
    print("Special tokens:")
    for tok in tokenizer.special_tokens:
        print(f"{tok}: {tokenizer.stoi[tok]}")
