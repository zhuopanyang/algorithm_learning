# -*- coding: utf-8 -*
"""
98. 所有可达路径
题目描述
给定一个有 n 个节点的有向无环图，节点编号从 1 到 n。请编写一个函数，找出并返回所有从节点 1 到节点 n 的路径。
每条路径应以节点编号的列表形式表示。

输入描述
第一行包含两个整数 N，M，表示图中拥有 N 个节点，M 条边
后续 M 行，每行包含两个整数 s 和 t，表示图中的 s 节点与 t 节点中有一条路径

输出描述
输出所有的可达路径，路径中所有节点之间空格隔开，每条路径独占一行，存在多条路径，路径输出的顺序可任意。如果不存在任何一条路径，则输出 -1。
注意输出的序列中，最后一个节点后面没有空格！ 例如正确的答案是 `1 3 5`,而不是 `1 3 5 `， 5后面没有空格！

输入示例
5 5
1 3
3 5
1 2
2 4
4 5
输出示例
1 3 5
1 2 4 5
"""
from collections import defaultdict


class Solution:
    def __init__(self):
        """
        初始化函数
        """
        self.res = []
        self.path = []

    def dfs(self, graph: list, start_index: int, n: int) -> None:
        # 终止条件
        if start_index == n:
            self.res.append(self.path.copy())
            return

        # 1) 邻接矩阵的形式，开始遍历
        for i in range(1, n + 1):
            if graph[start_index][i] == 1:
                self.path.append(i)
                self.dfs(graph, i, n)
                self.path.pop()

        # # 2) 邻接矩阵的形式，开始遍历
        # for i in graph[start_index]:
        #     self.path.append(i)
        #     self.dfs(graph, i, n)
        #     self.path.pop()

    def main(self) -> list:
        """
        获得从1开始，到最后的城市n，所有的可能的路径
        :return: 返回所有可能的路径
        """
        # # ACM格式获得输入
        # n, m = map(int, input().split())
        #
        # # 1）邻接矩阵的格式
        # # graph = [[0] * (n + 1) for _ in range(n + 1)]
        #
        # # 2）邻接表的格式
        # graph = defaultdict(list)
        #
        # # 遍历
        # for _ in range(m):
        #     s, t = map(int, input().split())
        #     # 1）邻接矩阵的格式
        #     # graph[s][t] = 1
        #     # 2）邻接表的格式
        #     graph[s].append(t)

        n, m = 5, 5
        graph = [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
        ]

        # 初始化self.res，会存在一个初始化点1
        self.res = []
        self.path.append(1)

        self.dfs(graph, 1, n)
        return self.res


if __name__ == "__main__":
    test = Solution()
    res = test.main()

    if not res:
        print(-1)
    else:
        for path in res:
            print(" ".join(map(str, path)))
