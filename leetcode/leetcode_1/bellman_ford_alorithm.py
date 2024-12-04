# -*- coding: utf-8 -*
"""
题目描述
某国为促进城市间经济交流，决定对货物运输提供补贴。共有 n 个编号为 1 到 n 的城市，通过道路网络连接，
网络中的道路仅允许从某个城市单向通行到另一个城市，不能反向通行。

网络中的道路都有各自的运输成本和政府补贴，道路的权值计算方式为：运输成本 - 政府补贴。权值为正表示扣除了政府补贴后运输货物仍需支付的费用；
权值为负则表示政府的补贴超过了支出的运输成本，实际表现为运输过程中还能赚取一定的收益。

请找出从城市 1 到城市 n 的所有可能路径中，综合政府补贴后的最低运输成本。如果最低运输成本是一个负数，它表示在遵循最优路径的情况下，运输过程中反而能够实现盈利。

城市 1 到城市 n 之间可能会出现没有路径的情况，同时保证道路网络中不存在任何负权回路。

输入描述
第一行包含两个正整数，第一个正整数 n 表示该国一共有 n 个城市，第二个整数 m 表示这些城市中共有 m 条道路。
接下来为 m 行，每行包括三个整数，s、t 和 v，表示 s 号城市运输货物到达 t 号城市，道路权值为 v （单向图）。

输出描述
如果能够从城市 1 到连通到城市 n， 请输出一个整数，表示运输成本。如果该整数是负数，则表示实现了盈利。
如果从城市 1 没有路径可达城市 n，请输出 "unconnected"。

输入示例
6 7
5 6 -2
1 2 1
5 3 1
2 5 2
2 4 -3
4 6 4
1 3 5

输出示例
1

提示信息
示例中最佳路径是从 1 -> 2 -> 5 -> 6，路上的权值分别为 1 2 -2，最终的最低运输成本为 1 + 2 + (-2) = 1。

示例 2：
4 2
1 2 -1
3 4 -1
在此示例中，无法找到一条路径从 1 通往 4，所以此时应该输出 "unconnected"。

数据范围：
1 <= n <= 1000；
1 <= m <= 10000;
-100 <= v <= 100;
"""

from collections import deque, defaultdict


def bellman_ford_optimize(edges: list, n: int, start: int, end: int) -> int:
    """
    bellam_ford的优化算法：边的权值存在服
    :param edges:
    :param n:
    :param start:
    :param end:
    :return:
    """
    grid = defaultdict(list)
    for x, y, k in edges:
        grid[x].append([y, k])

    min_dist = [float("inf")] * (n + 1)
    min_dist[start] = 0

    visited = [False] * (n + 1)
    visited[start] = True

    queue = deque([start])

    while queue:
        cur = queue.popleft()
        visited[cur] = False

        for y, k in grid[cur]:
            if min_dist[cur] != float("inf") and min_dist[cur] + k < min_dist[y]:
                min_dist[y] = min_dist[cur] + k
                if visited[y] == False:
                    queue.append(y)
                    visited[y] = True

    if min_dist[end] == float("inf"):
        return -1
    else:
        return min_dist[end]


def bellman_ford(edges, n, start, end):
    min_dist = [float("inf")] * (n + 1)
    min_dist[start] = 0

    # 松弛 n - 1次
    for i in range(1, n):
        update = False

        for x, y, k in edges:
            if min_dist[x] != float("inf") and min_dist[x] + k < min_dist[y]:
                min_dist[y] = min_dist[x] + k
                update = True

        if update is False:
            break

    if min_dist[end] == float("inf"):
        return -1
    else:
        return min_dist[end]


if __name__ == '__main__':
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        x, y, k = map(int, input().split())
        edges.append([x, y, k])

    start, end = 1, n

    res = bellman_ford_optimize(edges, n, start, end)
    print(res)
