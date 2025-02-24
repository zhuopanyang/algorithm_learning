# -*- coding: utf-8 -*
"""

"""

from collections import defaultdict


class UnionFind():
    """
    定义一个并查集
    """
    def __init__(self, size):
        self.parent = list(range(size + 1))

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            self.parent[root_u] = root_v

    def is_same(self, u, v):
        return self.find(u) == self.find(v)


def is_tree_sfter_remove_edge(edges, edge, n):
    uf = UnionFind(n)

    for i in range(n):
        if i == edge:
            continue

        u, v = edges[i]
        if uf.is_same(u, v):
            return False
        else:
            uf.union(u, v)
    return True


def get_remove_edge(edges, n):
    uf = UnionFind(n)

    for i in range(n):
        u, v = edges[i]
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

    vec = list()
    for i in range(n - 1, -1, -1):
        u, v = edges[i]
        if in_degree[v] == 2:
            vec.append(i)

    # 输出
    if len(vec) > 0:
        if is_tree_sfter_remove_edge(edges, vec[0], n):
            print(edges[vec[0]][0], edges[vec[0]][1])
        else:
            print(edges[vec[1]][0], edges[vec[1]][1])
    else:
        get_remove_edge(edges, n)
