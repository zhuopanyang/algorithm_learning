# -*- coding: utf-8 -*
"""
题目：求累积最大的连续子数组的结果:
给定 n 个存在正、负整数组成的数组，求最长的连续子串，其累乘最大。

求解：
要求解最长连续子串的累乘最大值问题，我们需要考虑乘法的一些特性，特别是负数的乘积。
一个负数乘以一个负数会得到一个正数，因此，我们需要跟踪当前子串的累乘积，同时也要跟踪包含一个负数的子串的累乘积。

"""


def max_product_subarray(nums: list) -> int:
    """
    求累积最大的连续子数组的结果
    :param nums:    输入的数组
    :return:    返回一个最大的连续子数组的累积结果
    """
    # 边界条件
    if len(nums) == 0:
        return 0

    # 初始化最大值和当前值
    max_product = nums[0]
    curr_product = nums[0]
    # 初始化最小值和当前值
    min_product = nums[0]
    curr_min = nums[0]

    # 遍历数组
    for i in range(1, len(nums)):
        # 如果当前数字是负数，需要交换curr_product和curr_min
        if nums[i] < 0:
            curr_product, curr_min = curr_min, curr_product

        # 更新当前的乘积和最小乘积
        curr_product = max(nums[i], curr_product * nums[i])
        curr_min = min(nums[i], curr_min * nums[i])

        # 更新全局最大乘积
        max_product = max(max_product, curr_product)

    return max_product


if __name__ == '__main__':
    # 示例
    nums = [2, 3, -2, 4, 1, -3]
    print(max_product_subarray(nums))  # 输出应该是 6，对应的子数组是 [2, 3]
