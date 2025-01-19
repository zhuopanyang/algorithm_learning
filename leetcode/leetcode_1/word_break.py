# -*- coding: utf-8 -*
"""
Leetcode 139.单词拆分
给你一个字符串 s 和一个字符串列表 wordDict 作为字典。
如果可以利用字典中出现的一个或多个单词拼接出 s 则返回 true。

注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。
"""


def wordBreak(s: str, wordDict: list[str]) -> bool:
    """
    返回是否可以找到合并成 s 的排列，是则返回true
    【动态规划】
    排列数：先遍历背包容量，后遍历物品
    完全背包：顺序遍历背包容量
    :param s:   输入的字符串
    :param wordDict:    输入的字符数组
    :return:    返回是否可以
    """
    # 定义并初始化dp数组
    dp = [float("inf")] * (len(s) + 1)
    dp[0] = 0

    # 开始遍历
    for j in range(1, len(s) + 1):
        for word in wordDict:
            if j >= len(word) and word == s[j - len(word): j]:
                dp[j] = min(dp[j], dp[j - len(word)] + 1)

    # 返回结果
    return dp[len(s)] != float("inf")


if __name__ == '__main__':
    s = "leetcode"
    word_dict = ["leet", "code"]

    print(wordBreak(s, word_dict))