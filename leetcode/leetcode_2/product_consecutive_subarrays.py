# -*- coding: utf-8 -*
"""
【滑动窗口】
Leetcode 713
给你一个整数数组 nums 和一个整数 k ，请你返回子数组内所有元素的乘积严格小于 k 的连续子数组的数目。

示例 1：
输入：nums = [10,5,2,6], k = 100
输出：8

解释：8 个乘积小于 100 的子数组分别为：[10]、[5]、[2]、[6]、[10,5]、[5,2]、[2,6]、[5,2,6]。
需要注意的是 [10,5,2] 并不是乘积小于 100 的子数组。

"""


def numSubarrayProductLessThanK(nums: list, k: int) -> int:
    """
    给你一个整数数组 nums 和一个整数 k ，请你返回子数组内所有元素的乘积严格小于 k 的连续子数组的数目。
    :param nums:    输入的数组
    :param k:   乘积k
    :return:    返回符合的子数组的数目
    """
    # 边界判断
    if k <= 1:
        return 0

    left = 0
    cur_prod = 1
    res = 0

    # 不断遍历右指针
    for right in range(len(nums)):
        # 不断积累，滑动窗口内的乘积
        cur_prod *= nums[right]

        # 如果当前的乘积 >= k，则收缩窗口
        while cur_prod >= k:
            cur_prod /= nums[left]
            left += 1

        # 计算以right结尾的子数组的数量（这个意图，保持子数组不会重复）
        res += right - left + 1  # 并累加起来

    # 返回结果
    return res


if __name__ == '__main__':
    nums = [10, 5, 2, 6]
    k = 100

    res = numSubarrayProductLessThanK(nums, k)
    print(res)
