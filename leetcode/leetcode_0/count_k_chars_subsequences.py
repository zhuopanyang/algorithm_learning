# -*- coding: utf-8 -*
"""
有一个字符串，长度为n，每个字符都是a-z之间的字符，
定义: 子字符串的含义是去掉原字符串的某些字符后，顺序保留不变形成的新字符串就是子字符串，
请问有多少个子字符串，其字符种类数量是k个

理论分析
给定一个字符串，问题要求找到字符种类数量为 kkk 的所有可能子序列的数量。由于子序列考虑的是字符的顺序，而非连续，因此，这涉及到组合数学。

步骤如下：
字符频率统计： 统计字符串中每个字符出现的频率。
选择字符种类组合： 从统计中选择构成子序列的恰好 kkk 种不同的字符。对于选定的每一种字符，在其可能的数量范围内选择子集参与子序列组合形成。
子序列组合计算： 对于选定的 kkk 种字符，每一种可以从其频率中选择若干个构成顺序不变的子序列。

"""

from itertools import combinations
from collections import Counter


def count_k_distinct_subsequences(s: str, k: int) -> int:
    """
    计算字符串中的符合条件的子字符串数量，条件是存在k个种类的字符
    :param s:   输入的字符串
    :param k:   需要包含k种字符
    :return:    返回符合条件的子字符串数量
    """
    # 统计每个字符的频率
    frequency = Counter(s)
    unique_chars = list(frequency.keys())

    # 如果种类数小于k，显然不能满足条件，直接返回0
    if len(unique_chars) < k:
        return 0

    total_count = 0

    # 对字符种类进行选择组合
    for chosen_chars in combinations(unique_chars, k):
        # 计算形成子序列的组合数量
        sub_count = 1
        for char in chosen_chars:
            # 对于每种字母，所有非空子集数目是2^出现次数 - 1（去掉空集）
            char_count = frequency[char]
            sub_count *= (2 ** char_count - 1)

        total_count += sub_count

    return total_count


# 示例用法
s = "eeabcd"
k = 5
print(count_k_distinct_subsequences(s, k))