# -*- coding: utf-8 -*
"""
输入一个旋转数组，如[5, 6, 7, 9, 10, 1, 2, 3, 4]，从中找到target数值，如果找到则返回索引值，找不到则返回-1
"""


def binary_search(rotated_nums: list, target: int) -> int:
    """
    使用二分查找法，来找到对应的target的索引位置
    :param rotated_nums:    旋转数组
    :param target:  需要找到的target
    :return:    返回target在旋转数组的索引位置
    """
    # 第一种区间 [)解法
    # left, right = 0, len(rotated_nums)
    #
    # while left < right:    # 使用[)区间
    #     mid = left + (right - left) // 2
    #     # 此时如果找到target，直接返回mid的索引
    #     if rotated_nums[mid] == target:
    #         return mid
    #
    #     # 开始二分查找
    #     # 此时，mid左侧是递增序列，右侧则有可能是非递增序列
    #     if rotated_nums[mid] > rotated_nums[0]:
    #         # 判断target在mid的左侧，还是右侧
    #         if rotated_nums[mid] >= target >= rotated_nums[0]:
    #             right = mid
    #         else:
    #             left = mid + 1
    #     else:   # 此时，mid右侧侧是递增序列，左侧则有可能是非递增序列
    #         # 判断target在mid的左侧，还是右侧
    #         if rotated_nums[mid] <= target <= rotated_nums[len(rotated_nums) - 1]:
    #             left = mid + 1
    #         else:
    #             right = mid

    # 第二种区间 [] 解法
    left, right = 0, len(rotated_nums) - 1

    while left <= right:  # 使用[] 区间
        mid = left + (right - left) // 2
        # 此时如果找到target，直接返回mid的索引
        if rotated_nums[mid] == target:
            return mid

        # 开始二分查找
        # 此时，mid左侧是递增序列，右侧则有可能是非递增序列
        if rotated_nums[mid] > rotated_nums[0]:
            # 判断target在mid的左侧，还是右侧
            if rotated_nums[mid] >= target >= rotated_nums[0]:
                right = mid - 1
            else:
                left = mid + 1
        else:   # 此时，mid右侧侧是递增序列，左侧则有可能是非递增序列
            # 判断target在mid的左侧，还是右侧
            if rotated_nums[mid] <= target <= rotated_nums[len(rotated_nums) - 1]:
                left = mid + 1
            else:
                right = mid - 1

    return -1


def binary_search_min(rotated_nums: list) -> int:
    """
    找出旋转数组中的 最小值 min_value
    :param rotated_nums:    旋转数组
    :return:    返回旋转数组中的最小值
    """
    left, right = 0, len(rotated_nums) - 1
    res = float("inf")

    # 二分查找法 [] 解法
    while left <= right:
        mid = left + (right - left) // 2
        cur = rotated_nums[mid]
        res = min(res, cur)

        # 二分查找
        if cur > rotated_nums[0]:
            left = mid + 1
        else:
            right = mid - 1

    return res


if __name__ == '__main__':
    rotated_nums = [5, 6, 7, 9, 10, 1, 2, 3, 4]
    target = 7
    target_index = binary_search(rotated_nums, target)
    print(target_index)
