# -*- coding: utf-8 -*
"""
给定字符串如，求出所有可能的拆分方式，使得每个元素都是回文子串

比如字符串如“aab”，可以拆分为["a", "a", "b"]或者["aa", "b"]

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


if __name__ == '__main__':
    s = "aabb"

    res = partition_palindromes(s)

    print(res)
