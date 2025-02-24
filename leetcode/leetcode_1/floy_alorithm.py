# -*- coding: utf-8 -*
"""
97. 小明逛公园
【多源，可正向、也可负向】
题目描述
小明喜欢去公园散步，公园内布置了许多的景点，相互之间通过小路连接，小明希望在观看景点的同时，能够节省体力，走最短的路径。


给定一个公园景点图，图中有 N 个景点（编号为 1 到 N），以及 M 条双向道路连接着这些景点。每条道路上行走的距离都是已知的。


小明有 Q 个观景计划，每个计划都有一个起点 start 和一个终点 end，表示他想从景点 start 前往景点 end。由于小明希望节省体力，他想知道每个观景计划中从起点到终点的最短路径长度。 请你帮助小明计算出每个观景计划的最短路径长度。

输入描述
第一行包含两个整数 N, M, 分别表示景点的数量和道路的数量。

接下来的 M 行，每行包含三个整数 u, v, w，表示景点 u 和景点 v 之间有一条长度为 w 的双向道路。

接下里的一行包含一个整数 Q，表示观景计划的数量。

接下来的 Q 行，每行包含两个整数 start, end，表示一个观景计划的起点和终点。

输出描述
对于每个观景计划，输出一行表示从起点到终点的最短路径长度。如果两个景点之间不存在路径，则输出 -1。
输入示例
7 3
2 3 4
3 6 6
4 7 8
2
2 3
3 4
输出示例
4
-1
提示信息
从 2 到 3 的路径长度为 4，3 到 4 之间并没有道路。

1 <= N, M, Q <= 1000.

1 <= w <= 10000.
"""


def floy(n: int, m: int) -> None:
    """
    【多源，可以正向、也可以负向】floy算法
    :param n:   点的数量
    :param m:   边的数量
    :return:    None
    """
    max_int = 10005
    # 定义三维dp数组，grid[i][j][k]，表示 节点i 到 节点j 以[1...k] 集合中的一个节点为中间节点的最短距离
    grid = [[[max_int] * (n + 1) for _ in range(n + 1)] for _ in range(n + 1)]
    # 初始化dp数组
    for _ in range(m):
        x, y, w = map(int, input().split())
        grid[x][y][0] = w
        grid[y][x][0] = w

    # 开始floyd算法，三层循环，先遍历k，接着遍历i / j（二者顺序可以随意）
    for k in range(1, n + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                # 递推公式：从节点 i 到节点 j 的最短路径，有两种情况：
                #（1）grid[i][j][k - 1]：不经过节点 k
                #（2）grid[i][k][k - 1] + grid[k][j][k - 1]：经过节点 k
                grid[i][j][k] = min(grid[i][j][k - 1], grid[i][k][k - 1] + grid[k][j][k - 1])

    # 输出结果
    z = int(input())
    for _ in range(z):
        start, end = map(int, input().split())
        if grid[start][end][n] == max_int:
            print(-1)
        else:
            print(grid[start][end][n])


if __name__ == '__main__':
    n, m = map(int, input().split())
    # 调用floy算法
    floy(n, m)
