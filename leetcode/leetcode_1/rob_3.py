# -*- coding: utf-8 -*
"""
小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为 root 。

除了 root 之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。
如果 两个直接相连的房子在同一天晚上被打劫 ，房屋将自动报警。

给定二叉树的 root 。返回 在不触动警报的情况下 ，小偷能够盗取的最高金额 。

输入: root = [3,2,3,null,3,null,1]
输出: 7
解释: 小偷一晚能够盗取的最高金额 3 + 3 + 1 = 7

"""


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    # 【动态规划】
    def rob(self, root:TreeNode) -> int:
        # dp数组（dp table）以及下标的含义：
        # 1. 下标为 0 记录 **不偷该节点** 所得到的的最大金钱
        # 2. 下标为 1 记录 **偷该节点** 所得到的的最大金钱
        dp = self.traversal(root)
        return max(dp)

    # 需要用到后序遍历，因为要通过递归函数的返回值来做下一步计算
    def traversal(self, root):
        # 递归终止条件
        if not root:
            return (0, 0)

        left = self.traversal(root.left)
        right = self.traversal(root.right)

        # 不偷当前节点，偷子节点
        val_0 = max(left[0], left[1]) + max(right[0], right[1])

        # 偷当前节点，不偷子节点
        val_1 = root.val + left[0] + right[0]

        # 返回结果
        return (val_0, val_1)
