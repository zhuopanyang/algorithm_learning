# -*- coding: utf-8 -*
"""
【动态规划算法】
有一个正整数数组，
（1）求最长的非递增子序列的长度；
（2）求可以组成的最少的递减子序列的个数。（可以转化为，求最长的非递减子序列的长度）

nums = [389, 207, 155, 300, 299, 170, 158, 65, 100, 60]
res_1 = 7
res_2 = 2

"""


def main(nums: list) -> list:
    """
    利用动态规划算法，来求解两个答案
    :param nums:   输入的数组
    :return:    返回结果
    """
    # (1) 求最长的非递增子序列的长度；
    # 定义并初始化dp数组
    res_1 = 0
    dp_1 = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(0, i):

            if nums[j] >= nums[i]:
                dp_1[i] = max(dp_1[i], dp_1[j] + 1)

        res_1 = max(res_1, dp_1[i])

    # （2）求可以组成的最少的递减子序列的个数。（可以转化为，求最长的非递减子序列的长度）
    # 定义并初始化dp数组
    res_2 = 0
    dp_2 = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(0, i):

            if nums[j] <= nums[i]:
                dp_2[i] = max(dp_2[i], dp_2[j] + 1)

        res_2 = max(res_2, dp_2[i])

    # 返回结果
    return res_1, res_2

if __name__ == '__main__':
    nums = [389, 207, 155, 300, 299, 170, 158, 65, 100, 60]

    res_1, res_2 = main(nums)
    print(res_1, res_2)
