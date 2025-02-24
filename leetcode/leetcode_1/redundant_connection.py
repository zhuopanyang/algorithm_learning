# -*- coding: utf-8 -*
"""
题目描述
有一种有向树,该树只有一个根节点，所有其他节点都是该根节点的后继。该树除了根节点之外的每一个节点都有且只有一个父节点，而根节点没有父节点。
有向树拥有 n 个节点和 n - 1 条边。如图：

现在有一个有向图，有向图是在有向树中的两个没有直接链接的节点中间添加一条有向边。如图：

输入一个有向图，该图由一个有着 n 个节点(节点编号 从 1 到 n)，n 条边，请返回一条可以删除的边，使得删除该条边之后该有向图可以被当作一颗有向树。

输入描述
第一行输入一个整数 N，表示有向图中节点和边的个数。

后续 N 行，每行输入两个整数 s 和 t，代表这是 s 节点连接并指向 t 节点的单向边

输出描述
输出一条可以删除的边，若有多条边可以删除，请输出标准输入中最后出现的一条边。
输入示例
3
1 2
1 3
2 3
输出示例
2 3

提示信息
在删除 2 3 后有向图可以变为一棵合法的有向树，所以输出 2 3

数据范围：
1 <= N <= 1000.
"""

from collections import defaultdict


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

    def find(self, u: int) -> None:
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


def is_tree_after_remove_edge(edges: list, edge: int, n: int) -> bool:
    """
    判断删除一条边edge后，是否为连通树
    :param edges:   所有的边集合
    :param edge:    需要删除的边
    :param n:   所有顶点的数量
    :return:    判断是否为连通树
    """
    uf = UnionFind(n)

    for i in range(len(edges)):
        if i == edge:
            continue
        u, v = edges[i]
        if uf.is_same(u, v):
            return False
        else:
            uf.union(u, v)

    return True


def get_remove_edge(edges: list, n: int) -> None:
    """
    已确定图中存在有向环，找到需要删除的那条边
    :param edges:   所有的边的集合
    :param n:   所有顶点的数量
    :return:    None
    """
    uf = UnionFind(n)

    for u, v in edges:
        if uf.is_same(u, v):
            print(u, v)
            return
        else:
            uf.union(u, v)


if __name__ == '__main__':
    n = int(input())
    edges = []
    in_degree = defaultdict(int)

    for i in range(n):
        u, v = map(int, input().split())
        in_degree[v] += 1
        edges.append([u, v])

    # 寻找入度威2的边，并记录其下标
    vec = list()
    for i in range(n - 1, -1, -1):
        if in_degree[edges[i][1]] == 2:
            vec.append(i)

    # 输出
    if len(vec) > 0:
        # 情况一： 删除输出顺序靠后的边
        if is_tree_after_remove_edge(edges, vec[0], n):
            print(edges[vec[0]][0], edges[vec[0]][1])
        else:
            # 情况二：只能删除特定的边
            print(edges[vec[1]][0], edges[vec[1]][1])
    else:
        # 情况三，存在环
        get_remove_edge(edges, n)
