# -*- coding: utf-8 -*
"""
leetcode 18 四数之和

给你一个由 n 个整数组成的数组 nums ，和一个目标值 target 。
请你找出并返回满足下述全部条件且不重复的四元组:
[nums[a], nums[b], nums[c], nums[d]] （若两个四元组元素一一对应，则认为两个四元组重复）：

0 <= a, b, c, d < n
a、b、c 和 d 互不相同
nums[a] + nums[b] + nums[c] + nums[d] == target
你可以按 任意顺序 返回答案 。
"""


def four_sum(nums: list, target: int) ->list:
    res = []
    nums.sort()

    # 开始遍历
    for i in range(len(nums)):
        # 1-剪枝
        if nums[i] > 0 and nums[i] > target:
            break

        # 2-去除重复的数字
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        # 此时，已经选好了数字 nums[i], 开始选第二个数字
        for j in range(i + 1, len(nums)):
            # 1-剪枝
            if nums[i] > 0 and nums[j] > 0 and nums[i] + nums[j] > target:
                break

            # 2-去除重复的数字
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue

            # 此时，已经选好了两个数字，nums[i]、nums[j]，还需要再选两个数字，使用双指针来查找
            left, right = j + 1, len(nums) - 1
            while left < right:
                cur_sum = nums[i] + nums[j] + nums[left] + nums[right]
                if cur_sum > target:
                    right -= 1
                elif cur_sum < target:
                    left += 1
                else:
                    # 此时，寻找到一个符合的四数组合
                    res.append([nums[i], nums[j], nums[left], nums[right]])
                    # 去除left、right两端重复的数字
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    # 最后跳出这个符合条件的组合
                    left += 1
                    right -= 1

    # 返回结果
    return res


if __name__ == '__main__':
    nums = [1, 0, -1, 0, -2, 2]
    target = 0
    res = four_sum(nums, target)
    print(res)
