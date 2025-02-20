# -*- coding: utf-8 -*
"""
给定字符串如，求出所有可能的拆分方式，使得每个元素都是回文子串

比如字符串如“aab”，可以拆分为["a", "a", "b"]或者["aa", "b"]

【回溯方法】 / 【dp算法】
"""


def is_palindrome(s: str) -> bool:
    """
    检查一个字符串是否是回文
    :param s: 输入的字符串
    :return:    返回bool，判断是否为回文子串
    """
    return s == s[::-1]


def backtrack(s: str, start: int, path: list, result: list) -> list:
    """
    回溯方法
    :param s: 输入的字符串
    :param start:   字符串开始的位置
    :param path:    当前的拆分集合
    :param result:  所有的回文子串拆分组合
    :return:    返回所有的组合
    """
    # 如果起始位置已经到达字符串末尾，说明找到了一种拆分方案
    if start == len(s):
        result.append(path[:])
        return

    # 尝试从start开始的所有可能的子串
    for end in range(start + 1, len(s) + 1):
        # 如果当前子串是回文，将其加入路径，并继续递归寻找下一个子串
        if is_palindrome(s[start:end]):
            path.append(s[start:end])
            backtrack(s, end, path, result)
            path.pop()


def partition_palindromes(s: str) -> list:
    """
    找到所有可能的回文子串拆分
    :param s:   输入的字符串
    :return:    返回所有的子串组合
    """
    result = []
    backtrack(s, 0, [], result)
    return result

def dp_solution(s: str) -> list:
    """
    用dp的解法，来输出所有可能的回文字符子串
    :param s:
    :return:
    """
    # 定义一个 dp 数组， dp[i][j] 表示从i到j之间的字符串为回文串
    dp = [[False] * len(s) for _ in range(len(s))]
    # 进行初始化
    for i in range(len(s)):
        dp[i][i] = True

    res = []

    # 开始遍历
    for i in range(len(s) - 1, -1, -1):
        for j in range(i, len(s)):
            # 判断当前 i 和 j的字符是否相等
            if s[i] == s[j]:
                # 当相等的时候，有两种情况下，dp[i][j]都是True的情况
                if j - i <= 1 or dp[i + 1][j - 1]:
                    dp[i][j] = True
                    res.append(s[i: j + 1])
            else:
                # 因为i和j所在的字符直接不相等，直接可以得出，该字符床不为回文子串
                dp[i][j] = False

    # 返回结果
    return res

if __name__ == '__main__':
    s = "aabb"

    # 回溯算法，输出的是所有可能的组合情况
    res_backtrack = partition_palindromes(s)
    print(res_backtrack)

    # dp算法，输出的是所有可能的回文子串的情况
    res_dp = dp_solution(s)
    print(res_dp)
