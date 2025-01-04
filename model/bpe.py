# -*- coding: utf-8 -*
"""
BPE分词的实现
"""

from collections import defaultdict, Counter


def get_status(vocab: Counter[str]) -> defaultdict[int]:
    """
    获得下一轮的组合情况
    :param vocab: 当前的分词表
    :return: 返回当前的组合情况
    """
    pairs = defaultdict(int)

    # 遍历分词表
    for word, count in vocab.items():
        symbols = word.split()
        # 遍历每一个词
        for i in range(len(symbols) - 1):
            # 按照顺序，不断组合新的组合
            pairs[(symbols[i], symbols[i + 1])] += count

    return pairs


def merge_vocab(pair: defaultdict[int], v_in: Counter[str]) -> dict:
    """
    将分词表中的pair进行合并
    :param pair:    当前最高频率的组合
    :param v_in:    当前的分词表
    :return:    返回新的分词表
    """
    v_out = {}
    # 当前频率最高的组合
    bigram = "".join(pair)
    # 遍历当前的分词表
    for word in v_in:
        # 不断进行替换
        new_word = word.replace(" ".join(pair), bigram)
        v_out[new_word] = v_in[word]

    return v_out


def bpe(text: list, num_iters: int) -> Counter[str]:
    """
    bpe分词： 基于统计的分词方法
    :param text: 输入的文本信息
    :param num_iters: 迭代的次数
    :return:    返回一个分词表
    """
    # 将每一个字符，单独计算
    vocab = Counter(" ".join(word) for word in text)

    # 开始10次合并最高频率的操作
    for i in range(num_iters):
        # 按照顺序，获得当前vocab的下一轮的所有组合
        pairs = get_status(vocab)
        if not pairs:
            break
        # 找出上述组合的频率最高的组合
        best_pair = max(pairs, key=pairs.get)
        # 将分词表中的该组合，进行合并
        vocab = merge_vocab(best_pair, vocab)
    return vocab


if __name__ == '__main__':
    text = ["low", "lower", "newest", "widest", "room", "rooms"]
    num_iters = 10

    # 利用bpe，进行分词操作
    vocab = bpe(text, num_iters)
    print(vocab)
