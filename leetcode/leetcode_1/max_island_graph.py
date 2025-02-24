# -*- coding: utf-8 -*
"""
题目描述
给定一个由 1（陆地）和 0（水）组成的矩阵，你最多可以将矩阵中的一格水变为一块陆地，在执行了此操作之后，矩阵中最大的岛屿面积是多少。

岛屿面积的计算方式为组成岛屿的陆地的总数。岛屿是被水包围，并且通过水平方向或垂直方向上相邻的陆地连接而成的。你可以假设矩阵外均被水包围。

输入描述
第一行包含两个整数 N, M，表示矩阵的行数和列数。之后 N 行，每行包含 M 个数字，数字为 1 或者 0，表示岛屿的单元格。

输出描述
输出一个整数，表示最大的岛屿面积。

输入示例
4 5
1 1 0 0 0
1 1 0 0 0
0 0 1 0 0
0 0 0 1 1

输出示例
6

"""
from collections import deque


# 定义一些全局变量
directions = [
    [0, 1],
    [0, -1],
    [1, 0],
    [-1, 0],
]
island_sizes = {}


def bfs(grid: list, x: int, y: int, key: int, cur_size: int):
    """
    广度优先遍历算法
    :param grid:    岛屿的图
    :param x:   当前位置的x
    :param y:   当前位置的y
    :param key: 当前岛屿的标记key
    :param cur_size:    当前标记key岛屿的面积大小
    :return:    返回面积大小
    """
    # 定义一个队列或者栈，来存放遍历到的元素
    que = deque([])
    que.append([x, y])

    # 处理当前节点
    cur_size += 1
    grid[x][y] = key

    while que:
        cur_x, cur_y = que.popleft()

        for direction in directions:
            next_x = cur_x + direction[0]
            next_y = cur_y + direction[1]

            if next_x < 0 or next_y < 0 or next_x >= len(grid) or next_y >= len(grid[0]):
                continue

            if grid[next_x][next_y] == 1:
                cur_size += 1
                grid[next_x][next_y] = key
                que.append([next_x, next_y])

    return cur_size


def dfs(grid: list, x: int, y: int, key: int, cur_size: int):
    """
    深度优先遍历算法
    :param grid:    岛屿的图
    :param x:   当前位置的x
    :param y:   当前位置的y
    :param key: 当前岛屿的标记key
    :param cur_size:    当前标记key岛屿的面积大小
    :return:    返回面积大小
    """
    # 终止条件
    if grid[x][y] != 1:
        return cur_size

    # 处理当前节点
    cur_size += 1
    grid[x][y] = key

    # 开始遍历四个方向
    for direction in directions:
        next_x = x + direction[0]
        next_y = y + direction[1]
        # 判断下一个位置，是否可以走
        if (next_x < 0 or next_y < 0 or next_x >= len(grid) or next_y >= len(grid[0])
                or grid[next_x][next_y] == 0):
            continue
        # 不断递归，深度遍历
        cur_size = dfs(grid, next_x, next_y, key, cur_size)

    # 返回结果
    return cur_size


def main():
    """
    主函数
    :return:
    """
    grid = []
    n, m = map(int, input().split())
    for i in range(n):
        row = list(map(int, input().split()))
        grid.append(row)

    # 先dfs遍历一遍，记录下来各个岛屿的面积大小
    key = 2     # 表示岛屿的key记号
    for i in range(n):
        for j in range(m):
            # 当遇到已经访问过的节点时候，直接跳过, 遇到海洋的时候，也直接跳过
            if grid[i][j] != 1:
                continue

            # 深度优先遍历
            cur_size = dfs(grid, i, j, key, 0)
            island_sizes[key] = cur_size
            key += 1

    # 遍历第二遍，寻找最适合放的位置
    res = 0
    flag = False
    for i in range(n):
        for j in range(m):
            # 终止条件，不断找为0的位置
            if grid[i][j] != 0:
                continue
            # 此时符合，将其设置为1
            flag = True
            tmp = 0
            used_islands = []   # 用来记录，已经收集过的岛屿的序号的列表
            for direction in directions:
                next_x = i + direction[0]
                next_y = j + direction[1]

                # 判断下一个位置，是否可以走
                if next_x < 0 or next_y < 0 or next_x >= n or next_y >= m:
                    continue
                if grid[next_x][next_y] in used_islands or grid[next_x][next_y] == 0:
                    continue

                # 此时，可知道下一个元素是一个没有收集过的岛屿
                tmp += island_sizes[grid[next_x][next_y]]
                used_islands.append(grid[next_x][next_y])
            tmp += 1
            res = max(res, tmp)

    # 判断一下，并输出最大的岛屿面积
    if flag:
        print(res)
    else:
        print(n * m)


if __name__ == '__main__':
    main()
