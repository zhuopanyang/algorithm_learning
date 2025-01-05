# -*- coding: utf-8 -*
"""
Wordpiece分词的实现
"""


from collections import defaultdict, Counter

sentences = [
    "我",
    "喜欢",
    "吃",
    "苹果",
    "他",
    "不",
    "喜欢",
    "吃",
    "苹果派",
    "I like to eat apples",
    "She has a cute cat",
    "you are very cute",
    "give you a hug",
]


def build_stats(sentences: list) -> defaultdict[int]:
    """
    将初始化的文本的每一个单词进行记录
    :param sentences:   输入的原始文本
    :return:    返回单词的记录
    """
    stats = defaultdict(int)
    for sentence in sentences:
        symbols = sentence.split()
        for symbol in symbols:
            stats[symbol] += 1
    return stats


def compute_pair_scores(stats: dict, splits: dict) -> dict:
    """
    获取每一种pair组合的概率
    :param stats: 初始文本的单词划分情况
    :param splits: 初始文本的词表划分情况
    :return:    返回每一种pair组合的概率情况
    """
    letter_freqs = defaultdict(int)
    pair_freqs = defaultdict(int)
    for word, freq in stats.items():
        split = splits[word]
        if len(split) == 1:
            letter_freqs[split[0]] += freq
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            letter_freqs[split[i]] += freq
            pair_freqs[pair] += freq
        letter_freqs[split[-1]] += freq

    # 统计出现的概率，并返回
    scores = {
        pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
        for pair, freq in pair_freqs.items()
    }
    return scores


def merge_pair(pair: list, stats: dict, splits: dict) -> dict:
    """
    将最大概率的pair，进行合并
    :param pair:    最大概率的pair组合
    :param stats:   单词表
    :param splits:  字符表
    :return:    返回最新的字符表
    """
    a, b = pair

    for word in stats:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                merge = a + b[2:] if b.startswith("##") else a + b
                split = split[:i] + [merge] + split[i + 2:]
            else:
                i += 1
        splits[word] = split
    return splits


if __name__ == '__main__':
    # 1-记录初始化的单词的数量
    stats = build_stats(sentences)
    print("stats: ", stats)

    # 2-将上述收集到的单词，细化到字符级
    alphabet = []
    for word in stats.keys():
        if word[0] not in alphabet:
            alphabet.append(word[0])
        for letter in word[1:]:
            if f"##{letter}" not in alphabet:
                alphabet.append(f"##{letter}")
    alphabet.sort()
    # 初始化词表
    vocab = alphabet.copy()
    print("vocab: ", vocab)
    # 根据原始文本，进行的词表划分情况
    splits = {
        word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
        for word in stats.keys()
    }

    # 3-形成pair对，获取每一种pair组合的概率
    pair_scores = compute_pair_scores(stats, splits)
    for i, key in enumerate(pair_scores.keys()):
        print(f"{key}: {pair_scores[key]}")
        if i >= 5:
            break

    # 找出最大概率的组合
    best_pair = ""
    max_score = None
    for pair, score in pair_scores.items():
        if max_score is None or max_score < score:
            best_pair = pair
            max_score = score
    print(best_pair, max_score)

    # 设置分词表的大小， 并开始不断遍历，形成最终的分词表
    vocab_size = 50
    while len(vocab) < vocab_size:
        # 找出最大概率的组合
        scores = compute_pair_scores(stats, splits)
        best_pair, max_score = "", None
        for pair, score in scores.items():
            if max_score is None or max_score < score:
                best_pair = pair
                max_score = score

        splits = merge_pair(best_pair, stats, splits)

        new_token = (
            best_pair[0] + best_pair[1][2:]
            if best_pair[1].startswith("##")
            else best_pair[0] + best_pair[1]
        )
        vocab.append(new_token)

    # 打印最后的分词表
    print("vocab: ", vocab)
