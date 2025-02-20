# -*- coding: utf-8 -*
"""
1.
求解杨辉三角
给定一个非负索引 rowIndex，返回「杨辉三角」的第 rowIndex 行。
在「杨辉三角」中，每个数是它左上方和右上方的数的和。
    1
   1 1
  1 2 1
 1 3 3 1
 1 4 6 4 1

 根据定义，是一个动态规划算法，递推公式为 C(n, k) = C(n-1, k-1) + C(n - 1, k)
 根据边界条件，任何形状的三角形边缘元素都是1，即 C(n, 0) = C(n, n) = 1

"""


def yanghui(rowIndex: int) -> list[int]:
    """
    生成杨辉三角的某一层的数字列表
    :param rowIndex:    输入杨辉三角最底的一层，是多少层
    :return:    返回一个杨辉三角形第 rowIndex 行的数据
    """
    # 初始化杨辉三角的第0行
    row = [1]

    # 生成从第1行到第rowIndex行
    for i in range(1, rowIndex + 1):
        # 新行重新开始为1
        new_row = [1] * (i + 1)
        # 填充新行的中间元素
        for j in range(1, i):
            new_row[j] = row[j - 1] + row[j]
        # 更新当前行
        row = new_row

    return row


if __name__ == '__main__':
    row_index = 4
    print(f"The {row_index}th row of Pascal's Triangle: {yanghui(row_index)}")
