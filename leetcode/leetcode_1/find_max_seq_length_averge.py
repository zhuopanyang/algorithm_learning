# -*- coding: utf-8 -*
"""
【将其转为，找出最长连续子数组，相加等于0】
题目：平均数为k的最长连续子数组:
给定 n 个正整数组成的数组，求平均数正好等于 k 的最长连续子数组的长度。

输入描述:
第一行输入两个正整数 n 和 k，用空格隔开。
第二行输入 n 个正整数 a_i，用来表示数组。
1 <= n <= 200,000
1 <= k, a_i <= 10^9

输出描述:
如果不存在任何一个连续子数组的平均数等于 k，则输出 -1。
否则输出平均数正好等于 k 的最长连续子数组的长度。

示例 1
输入:
5 2
1 3 2 4 1

输出
3

说明
取前三个数即可，平均数为 2。
"""


def find_max_seq_length_average(nums: list, n: int, k: int) -> int:
    """
    寻找平均数为k的最长连续子数组的长度
    :param nums: 输入的数组
    :param n: 数组的长度
    :param k: 平均数k
    :return: 返回一个最长的连续子数组的长度
    """
    # 用于记录最大的长度
    max_length = 0

    # 用于记录当前的前缀和长度
    prefix_sum = 0
    # 用于记录当前的前缀和的下一个位置的索引：left索引
    prefix_map = {0: 0}

    # 不断遍历
    for i in range(n):
        # 累计此时的前缀和
        prefix_sum += nums[i]
        # 累计此时的前缀和（由于k = 0）
        target = prefix_sum - k

        # 如果此时该前缀和，之前出现过，则表明之前的索引——目前的索引，所在的区间累计和为0，符合区间要求
        if target in prefix_map:
            cur_length = i - prefix_map[target] + 1
            max_length = max(max_length, cur_length)

        # 如果此时该前缀和，之前未出现过，则记录此下一个位置的索引，表示 left 值（第一次记录，必定是接近左端，符合最长的要求）
        if prefix_sum not in prefix_sum:
            prefix_map[prefix_sum] = i + 1

    # 返回结果
    return max_length if max_length != 0 else -1


if __name__ == '__main__':
    n, k = map(int, input().split())
    nums = list(map(int, input().split()))

    # 转换为相加为 0 的数组解决问题
    nums = [num - k for num in nums]

    res = find_max_seq_length_average(nums, n, k)
    print(res)
