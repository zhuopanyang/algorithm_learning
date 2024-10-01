# -*- coding: utf-8 -*
"""
Leetcode的第1531题. 压缩字符串 II
行程长度编码 是一种常用的字符串压缩方法，它将连续的相同字符（重复 2 次或更多次）替换为字符和表示字符计数的数字（行程长度）。
例如，用此方法压缩字符串 "aabccc" ，将 "aa" 替换为 "a2" ，"ccc" 替换为` "c3" 。因此压缩后的字符串变为 "a2bc3" 。
注意，本问题中，压缩时没有在单个字符后附加计数 '1' 。
给你一个字符串 s 和一个整数 k 。你需要从字符串 s 中删除最多 k 个字符，以使 s 的行程长度编码长度最小。
请你返回删除最多 k 个字符后，s 行程长度编码的最小长度 。

题解：
首先将题意转换成 从字符串中，选取 T=n−k 字符，使编码长度最小.
定义dp[p][cnt]:
    p: 从字符串的第p位置开始
    cnt: 当前已经选取了 cnt 个字符
"""


def cal_len(x):
    """
    计算字符重复x次压缩后的长度
    :param x:   输入的连续相同的字符串长度
    :return:    返回压缩后的字符的长度
    """
    if not x:
        return 0
    elif x == 1:
        return 1
    elif x < 10:
        return 2
    elif x < 100:
        return 3
    return 4


def remove_k_chars(s: str, k: int) -> int:
    """
    使用动态规划算法，删除当前字符中至多k个字符，保留后的子字符的最小的压缩字符长度
    :param s:   输入的原始的字符
    :param k:   可以删除至多k个字符
    :return:    返回完成后的最小的压缩字符串长度
    """
    n = len(s)
    t = n - k
    # 假如没有字符串可以取，直接返回0
    if not t:
        return 0

    # 1）定义dp数组
    # 二维数组 dp[p][cnt]
    #      p: 从字符串的第p位置开始
    #      cnt: 当前已经选取了 cnt 个字符
    dp = [[float("inf") for _ in range(t + 1)] for _ in range(n + 1)]
    # 2）默认初始化

    # 3）开始遍历，求的是组合数，先遍历t，后遍历n
    for j in range(1, t + 1):
        for i in range(n - 1, -1, -1):  # 从状态转移方程，可以知道 i 是需要从尾部遍历到头部的
            same = 0

            # case 1: 从左边开始选跟第一个（i位置）相同的字符
            for p in range(i, n):
                # 如果相同，则选出来
                if s[p] == s[i]:
                    same += 1
                # 假如该相同的字符数量，可以达到j的水平，直接进行比较，返回，跳出循环（因为循环再接下去是增大字符长度的）
                if same == j:
                    dp[i][j] = min(dp[i][j], cal_len(same))
                    break
                # 此时，进行状态转移方程变化
                # dp[i][j] = min(dp[i][j]，选取当前字符的same个字符后，接下来从p + 1的位置开始的情况)
                dp[i][j] = min(dp[i][j], cal_len(same) + dp[p + 1][j - same])

            # case 2: 不选第一个字符
            dp[i][j] = min(dp[i][j], dp[i + 1][j])

    # 返回结果
    return dp[0][t]


if __name__ == '__main__':
    s = "aaaccbbbaaa"
    k = 2

    min_compressed_len = remove_k_chars(s, k)
    print(min_compressed_len)
