# -*- coding: utf-8 -*
"""
并查集主要有三个功能：

寻找根节点，函数：find(int u)，也就是判断这个节点的祖先节点是哪个
将两个节点接入到同一个集合，函数：join(int u, int v)，将两个节点连在同一个根节点上
判断两个节点是否在同一个集合，函数：isSame(int u, int v)，就是判断两个节点是不是同一个根节点

107. 寻找存在的路径
题目描述
给定一个包含 n 个节点的无向图中，节点编号从 1 到 n （含 1 和 n ）。
你的任务是判断是否有一条从节点 source 出发到节点 destination 的路径存在。

输入描述
第一行包含两个正整数 N 和 M，N 代表节点的个数，M 代表边的个数。
后续 M 行，每行两个正整数 s 和 t，代表从节点 s 与节点 t 之间有一条边。
最后一行包含两个正整数，代表起始节点 source 和目标节点 destination。

输出描述
输出一个整数，代表是否存在从节点 source 到节点 destination 的路径。如果存在，输出 1；否则，输出 0。

输入示例
5 4
1 2
1 3
2 4
3 4
1 4
输出示例
1
"""


class UnionFind():
    """
    并查集的实现类
    """

    def __init__(self, size: int) -> None:
        """
        并查集的初始化函数
        :param size:
        """
        # 初始化并查集
        self.parents = list(range(size + 1))

    def find(self, u: int) -> int:
        """
        查找并查集中的某个元素的root节点
        :param u:   搜索的节点u
        :return:    返回该节点的根节点，这个过程中顺便进行路径压缩
        """
        if self.parents[u] != u:
            self.parents[u] = self.find(self.parents[u]) # 不断路径压缩
        return self.parents[u]

    def union(self, u: int, v: int) -> None:
        """
        将一条边 (u，v) 添加进来并查集中
        :param u:   边的一个节点
        :param v:   边的另一个节点
        :return:    None
        """
        root_u = self.find(u)
        root_v = self.find(v)
        # 检查一下边的两个节点的根节点是否一致，是的话，表示已经连通，不需要添加
        if root_u != root_v:
            self.parents[root_u] = root_v

    def is_same(self, u: int, v: int) -> bool:
        """
        判断两个节点是否存在共同的root节点
        :param u:   搜索的第一个节点
        :param v:   搜索的第二个节点
        :return:    返回是否对
        """
        return self.find(u) == self.find(v)


def main():
    N, M = map(int, input().split())

    uf = UnionFind(N)

    for i in range(M):
        s, t = map(int, input().split())
        uf.union(s, t)

    source, destination = map(int, input().split())

    if uf.is_same(source, destination):
        print(1)
    else:
        print(0)


if __name__ == "__main__":
    main()
