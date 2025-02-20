# -*- coding: utf-8 -*
"""
Leetcode的第84题. 柱状图中最大的矩形
给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。
求在该柱状图中，能够勾勒出来的矩形的最大面积。

# 【单调栈算法】
# 接雨水 (opens new window)是找每个柱子左右两边第一个大于该柱子高度的柱子
# 而本题是找每个柱子左右两边第一个小于该柱子的柱子

"""


def get_largest_rectangle_area(heights: list) -> int:
    """
    找每个柱子左右侧的第一个高度值小于该柱子的柱子
    单调栈：栈顶到栈底：从大到小（每插入一个新的小数值时，都要弹出先前的大数值）
    栈顶，栈顶的下一个元素，即将入栈的元素：这三个元素组成了最大面积的高度和宽度
    情况一：当前遍历的元素heights[i]大于栈顶元素的情况
    情况二：当前遍历的元素heights[i]等于栈顶元素的情况
    情况三：当前遍历的元素heights[i]小于栈顶元素的情况
    :param heights: 输入的柱状图的每个柱子的高度
    :return:    返回最大的矩形的面积
    """
    # 输入数组首尾各补上一个0（与42.接雨水不同的是，本题原首尾的两个柱子可以作为核心柱进行最大面积尝试
    heights.insert(0, 0)
    heights.append(0)

    res = 0
    # 单调栈中先放入第一个元素，单调栈中放的是索引值
    stack = [0]

    # 开始遍历
    for i in range(1, len(heights)):
        # 情况一：当前的元素 > 栈顶的元素
        if heights[i] > heights[stack[-1]]:
            stack.append(i)
        # 情况二：当前的元素 = 栈顶的元素
        elif heights[i] == heights[stack[-1]]:
            stack.pop()
            stack.append(i)
        # 情况三：当前的元素 < 栈顶的元素
        else:
            while stack and heights[i] < heights[stack[-1]]:
                mid_index = stack[-1]
                stack.pop()
                if stack:
                    left_index = stack[-1]
                    right_index = i
                    # 这个最大的矩形，很明显不包括 left_index 和 right_index 这两个边界的柱子
                    width = right_index - left_index - 1
                    height = heights[mid_index]
                    res = max(res, width * height)
            stack.append(i)

    return res


if __name__ == '__main__':
    heights = [2, 1, 5, 6, 2, 3]
    res = get_largest_rectangle_area(heights)
    print(res)
