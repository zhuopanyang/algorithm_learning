# -*- coding: utf-8 -*
"""
题目描述
在世界的某个区域，有一些分散的神秘岛屿，每个岛屿上都有一种珍稀的资源或者宝藏。国王打算在这些岛屿上建公路，方便运输。

不同岛屿之间，路途距离不同，国王希望你可以规划建公路的方案，如何可以以最短的总公路距离将 所有岛屿联通起来（注意：这是一个无向图）。

给定一张地图，其中包括了所有的岛屿，以及它们之间的距离。以最小化公路建设长度，确保可以链接到所有岛屿。

输入描述
第一行包含两个整数V 和 E，V代表顶点数，E代表边数 。顶点编号是从1到V。例如：V=2，一个有两个顶点，分别是1和2。

接下来共有 E 行，每行三个整数 v1，v2 和 val，v1 和 v2 为边的起点和终点，val代表边的权值。

输出描述
输出联通所有岛屿的最小路径总距离
输入示例
7 11
1 2 1
1 3 1
1 5 2
2 6 1
2 4 2
2 3 2
3 4 1
4 5 1
5 6 2
5 7 1
6 7 1
输出示例
6
提示信息
数据范围：
2 <= V <= 10000;
1 <= E <= 100000;
0 <= val <= 10000;

prim算法核心就是三步，我称为prim三部曲，大家一定要熟悉这三步，代码相对会好些很多：
第一步，选距离生成树最近节点
第二步，最近节点加入生成树
第三步，更新非生成树节点到生成树的距离（即更新minDist数组）

prim 算法是维护节点的集合，而 Kruskal 是维护边的集合。

Kruskal 与 prim 的关键区别在于，prim维护的是节点的集合，而 Kruskal 维护的是边的集合。
如果 一个图中，节点多，但边相对较少，那么使用Kruskal 更优。
而 prim 算法是对节点进行操作的，节点数量越少，prim算法效率就越优。
"""


class UnionFind():
    """
    定义并查集
    """

    def __init__(self, size: int) -> None:
        """
        初始化函数
        :param size:    初始化大小
        """
        self.parent = list(range(size + 1))

    def find(self, u: int) -> int:
        """
        寻找一个顶点的根节点
        :param u:   一个寻找的节点
        :return:    返回该节点的根节点
        """
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])

        return self.parent[u]

    def union(self, u: int, v: int) -> None:
        """
        将两个节点所形成的边，添加到并查集内
        :param u:   节点u
        :param v:   节点v
        :return:    None
        """
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            self.parent[root_u] = root_v

    def is_same(self, u: int, v: int) -> bool:
        """
        判断两个节点是否连通
        :param u:   节点u
        :param v:   节点v
        :return:    返回是否连通在一起
        """
        return self.find(u) == self.find(v)


def prim(v: int, edges: list) -> int:
    """
    prim算法，求联通所有顶点的最小路径的总距离，找点
    :param v:   各个顶点
    :param edges:   各个边
    :return:    返回最小路径的总距离
    """
    # 定义一个矩阵，表示各个边的权重信息
    grid = [[10001] * (v + 1) for _ in range(v + 1)]

    # 读取边的信息并开始填充矩阵
    for edge in edges:
        x, y, k = edge
        grid[x][y] = k
        grid[y][x] = k

    # 初始化一个，所有节点到最小生成树的最小距离
    min_dist = [10001] * (v + 1)

    # 记录节点是否在树里
    is_in_tree = [False] * (v + 1)

    # 【记录访问边】记录最后访问的边的集合
    visited_edges = [-1] * (v + 1)

    # prim算法主循环
    for i in range(1, v):
        cur = -1
        min_val = float("inf")

        # 选择距离生成树最近的节点
        for j in range(1, v + 1):
            if is_in_tree[j] is False and min_dist[j] < min_val:
                min_val = min_dist[j]
                cur = j

        # 将最近的节点加入生成树中
        is_in_tree[cur] = True

        # 更新非生成树节点到生成树的距离
        for j in range(1, v + 1):
            if is_in_tree[j] is False and grid[cur][j] < min_dist[j]:
                min_dist[j] = grid[cur][j]
                # 【记录访问边】记录最后访问的边的集合
                visited_edges[j] = cur

    # 统计结果，并返回
    res = sum(min_dist[2: v + 1])
    print(min_dist)
    return res


def kruskal(v: int, edges: list) -> int:
    """
    kruskal算法，，求联通所有顶点的最小路径的总距离，找边
    :param v:   各个顶点
    :param edges:   各个边
    :return:    返回最小路径的总距离
    """
    # 将边进行排序
    edges = sorted(edges, key=lambda edge: edge[2])

    uf = UnionFind(v)
    res = 0

    for edge in edges:
        x, y = uf.find(edge[0]), uf.find(edge[1])
        if x != y:
            res += edge[2]
            uf.union(edge[0], edge[1])

    return res


if __name__ == '__main__':
    v, e = map(int, input().split())

    edges = []
    index = 2

    for _ in range(e):
        x, y, k = map(int, input().split())
        edges.append((x, y, k))
        index += 3

    res = prim(v, edges)
    print(res)
