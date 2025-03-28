# -*- coding: utf-8 -*
"""
【单源正向】
题目描述
小明是一位科学家，他需要参加一场重要的国际科学大会，以展示自己的最新研究成果。
小明的起点是第一个车站，终点是最后一个车站。然而，途中的各个车站之间的道路状况、交通拥堵程度以及可能的自然因素（如天气变化）等不同，
这些因素都会影响每条路径的通行时间。
小明希望能选择一条花费时间最少的路线，以确保他能够尽快到达目的地。

输入描述
第一行包含两个正整数，第一个正整数 N 表示一共有 N 个公共汽车站，第二个正整数 M 表示有 M 条公路。
接下来为 M 行，每行包括三个整数，S、E 和 V，代表了从 S 车站可以单向直达 E 车站，并且需要花费 V 单位的时间。

输出描述
输出一个整数，代表小明从起点到终点所花费的最小时间。

输入示例
7 9
1 2 1
1 3 4
2 3 2
2 4 5
3 4 2
4 5 3
2 6 4
5 7 4
6 7 9

输出示例
12

提示信息
能够到达的情况：
如下图所示，起始车站为 1 号车站，终点车站为 7 号车站，绿色路线为最短的路线，路线总长度为 12，则输出 12。

不能到达的情况：
如下图所示，当从起始车站不能到达终点车站时，则输出 -1。

数据范围：
1 <= N <= 500;
1 <= M <= 5000;

最短路径算法：dijkstra算法
dijkstra 算法可以同时求 起点到所有节点的最短路径
权值不能为负数

这里我也给出 dijkstra三部曲：
第一步，选源点到哪个节点近且该节点未被访问过
第二步，该最近节点被标记访问过
第三步，更新非访问节点到源点的距离（即更新minDist数组）
"""

import heapq


class Edge:
    def __init__(self, to, val):
        self.to = to    # 邻接顶点
        self.val = val  # 边的权重


def dijkstra_optimize(edges: list, n: int, start: int, end: int) -> int:
    """
    【单源正边】dijkstra堆优化算法
    :param edges:   输入的边的集合
    :param n:   n个顶点数量
    :param start:   起点
    :param end:     终点
    :return:    返回最短路的总和
    """
    # 使用邻接表，来表示图
    grid = [[] for _ in range(n + 1)]
    for x, y, k in edges:
        grid[x].append(Edge(y, k))

    # 初始化距离数组、访问数组
    min_dist = [float("inf")] * (n + 1)
    visited = [False] * (n + 1)

    # 定义一个最小堆，并放入第一个元素
    pq = []
    heapq.heappush(pq, (0, start))  # 堆里面存放的是 (x, y)表示源点到y的距离是x
    min_dist[start] = 0

    # 判断当堆不为空的时候
    while pq:
        cur_dist, cur_node = heapq.heappop(pq)  # 压出最小堆的一个元素
        # 如果访问过，直接跳过
        if visited[cur_node]:
            continue
        # 设置当前节点
        visited[cur_node] = True

        for sub_edge in grid[cur_node]:
            # 三个条件：（1）当前节点未访问过、（2）当前节点和cur节点连接【该条件在当前邻接表结构上，默认成立】
            # （3）源点-cur节点 + cur节点-当前节点的距离 < minsDist[cur]
            if visited[sub_edge.to] is False and cur_dist + sub_edge.val < min_dist[sub_edge.to]:
                min_dist[sub_edge.to] = cur_dist + sub_edge.val
                heapq.heappush(pq, (min_dist[sub_edge.to], sub_edge.to))

    if min_dist[end] == float("inf"):
        return -1
    else:
        return min_dist[end]


def dijkstra(edges: list, n: int, start: int, end: int) -> int:
    """
    【单源正边】dijkstra（朴素版）算法
    :param edges:   输入的边的集合
    :param n:   n个顶点数量
    :param start:   起点
    :param end:     终点
    :return:    返回最短路的总和
    """
    # 初始化邻接矩阵
    grid = [[float("inf")] * (n + 1) for _ in range(n + 1)]
    for x, y, k in edges:
        grid[x][y] = k

    # 初始化距离数组和访问数组
    minDist = [float("inf")] * (n + 1)
    visited = [False] * (n + 1)

    # 初始化起点到自身的距离
    minDist[start] = 0

    # 开始遍历所有的节点
    for _ in range(1, n + 1):
        minVal = float("inf")
        cur = -1

        # 选择距离原点最近且未访问过的节点
        for v in range(1, n + 1):
            if visited[v] is False and minDist[v] < minVal:
                minVal = minDist[v]
                cur = v

        # 如果找不到未访问过的节点，则提前结束
        if cur == -1:
            break

        # 标记当前节点为访问过
        visited[cur] = True

        # 更新未访问过节点到原点的距离
        for v in range(1, n + 1):
            # 三个条件：（1）当前节点未访问过、（2）当前节点和cur节点连接
            # （3）源点-cur节点+cur节点-当前节点的距离 < minsDist[cur]
            if (visited[v] is False and grid[cur][v] != float("inf")
                    and minDist[cur] + grid[cur][v] < minDist[v]):
                minDist[v] = minDist[cur] + grid[cur][v]

    if minDist[end] == float("inf"):
        return -1
    else:
        return minDist[end]


if __name__ == '__main__':
    n, m = map(int, input().split())

    edges = []
    for _ in range(m):
        x, y, k = map(int, input().split())
        edges.append([x, y, k])

    start, end = 1, n

    res = dijkstra(edges, n, start, end)
    print(res)
