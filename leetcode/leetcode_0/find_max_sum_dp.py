# -*- coding: utf-8 -*
"""
用python解答，有一个int类型的数组，我们每次可以从中选出连续的三个数字，累加其中中间的元素，求最大值是多少，
用过的数字不能再用，且每次都要选出连续的三个数字才行。

边界条件：
判断数组长度是否小于 3，如果小于则不能选取三个元素，返回 0。

动态规划数组 (dp)：
dp[i] 用于存储到第 i 个元素能获得的最大和。
dp[2] 被初始化为 arr[1]，因为这是第一个可以计算的中间元素。

迭代计算：
从 i = 3 开始遍历到 n，在每一步计算当前 dp 值，
（1）可以选择当前的中间元素 arr[i-1]，并加上 dp[i-3]（之前的最佳值，确保不重叠）
（2）或者不选择当前的中间元素而继续保持最佳值 dp[i-1]。

更新最大值：
利用 max_value 变量来维护到目前为止最大的中间元素的和。

返回值：
最后返回 max_value，即为我们所求的结果。

"""


def max_sum_of_middle_elements(arr: list) -> int:
    """
    选出最大sum的组合的情况，并返回最大的数值sum
    :param arr: 输入的数组
    :return:    返回的最大的sum数值
    """
    n = len(arr)
    if n < 3:  # 如果数组长度小于3，不能选择三个元素
        return 0

    # 定义dp数组
    # dp[i]表示选择到第i个元素的最大和。
    dp = [0] * n

    # 初始化dp数组
    # 第一次选择三个元素时，可以直接计算。
    dp[2] = arr[1]

    # 维护一个最大值
    max_value = dp[2]

    # 从第四个元素开始到最后一个元素
    for i in range(3, n):
        # 可以选择(i-2, i-1, i)这三个元素，增加中间的元素
        dp[i] = max(dp[i - 1], dp[i - 3] + arr[i - 1])  # 选择中间元素时，不能重叠
        max_value = max(max_value, dp[i])

    return max_value


if __name__ == '__main__':
    # 示例用法
    arr = [1, 1, 4, 9, 1, 6, 1]
    result = max_sum_of_middle_elements(arr)
    print(f"最大值为：{result}")
