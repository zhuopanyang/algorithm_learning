# -*- coding: utf-8 -*
"""
有一个长度为n的正整数数组（数组中的元素可以重复），我们从中删除（n-k）个元素，剩下k个元素形成一个子序列.

让两个人A和B分别选取，A可以从左往右开始选，可以选取0到k个元素，需要连续的，
剩下的给B，因此，A会有一个子数组，B也有一个子数组。

他们各自的子数组的和，选出大的那个元素，该元素需要取到最小，请输出该数值。

1）子序列的选择：我们需要从原数组中选择一个长度为k的子序列。这个子序列可以通过删除n - k个元素得到。
2）A和B的选择：A可以从左往右选择连续的子数组，B则从剩下的部分中选择连续的子数组。我们需要找到一种选择方式，使得两者的和的最大值尽可能小。

解题思路
动态规划：我们可以使用动态规划来记录在选择子序列时的最优解。
贪心策略：在选择子序列时，我们需要贪心地选择当前最优的元素，以确保最终的最大值最小化。

算法设计
预处理：计算前缀和和后缀和，以便快速计算子数组的和。
动态规划表：维护一个二维数组dp[i][j]，表示前i个元素中选择j个元素的最大和。
贪心选择：在每一步选择时，选择当前最优的元素，以确保最终的最大值最小化。
"""


def min_max_sum(nums, k):
    n = len(nums)
    # 计算前缀和
    prefix_sum = [0] * (n + 1)
    for i in range(n):
        prefix_sum[i + 1] = prefix_sum[i] + nums[i]

    # 计算后缀和
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + nums[i]

    # 动态规划表，dp[i][j]表示前i个元素中选择j个元素的最大和
    dp = [[0] * (k + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, k + 1):
            dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - 1] + nums[i - 1])

    min_max = float('inf')
    # 遍历所有可能的分割点
    for i in range(k + 1):
        j = k - i
        if i == 0:
            a_sum = 0
        else:
            a_sum = dp[n][i]
        if j == 0:
            b_sum = 0
        else:
            b_sum = suffix_sum[n - j]
        current_max = max(a_sum, b_sum)
        if current_max < min_max:
            min_max = current_max

    return min_max


# 测试用例
nums = [8, 1, 10, 1, 1, 1]
k = 3
print(min_max_sum(nums, k))  # 输出 6
