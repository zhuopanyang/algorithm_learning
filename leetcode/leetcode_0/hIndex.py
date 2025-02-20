# -*- coding: utf-8 -*
"""
给你一个整数数组 citations ，其中 citations[i] 表示研究者的第 i 篇论文被引用的次数。计算并返回该研究者的 h 指数。
根据维基百科上 h 指数的定义：h 代表“高引用次数” ，一名科研人员的 h 指数 是指他（她）至少发表了 h 篇论文，并且每篇论文 至少 被引用 h 次。
如果 h 有多种可能的值，h 指数 是其中最大的那个。

def hIndex(citations: list[int]) -> int:

"""


def hIndex(citations: list[int]) -> int:
    """
    获得最大的 h
    :param citations: 上述定义的论文的数组
    :return:   返回最大的 h 数值
    """
    citations = sorted(citations)
    n = len(citations)

    # 最差的结果，就是排序后的第一个元素
    res = citations[0]

    # 从尾部开始遍历，不断找最大的可能的元素
    for i in range(n - 1, -1, -1):
        cur = citations[i]
        if cur <= n and citations[n - i - 1] >= cur:
            res = cur
            break

    return res


if __name__ == '__main__':
    citations = [1, 2, 3, 4]
    res = hIndex(citations)
    print(res)
