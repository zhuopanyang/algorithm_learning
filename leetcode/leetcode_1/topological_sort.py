# -*- coding: utf-8 -*
"""
题目描述
某个大型软件项目的构建系统拥有 N 个文件，文件编号从 0 到 N - 1，在这些文件中，某些文件依赖于其他文件的内容，这意味着如果文件 A 依赖于文件 B，
则必须在处理文件 A 之前处理文件 B （0 <= A, B <= N - 1）。请编写一个算法，用于确定文件处理的顺序。
输入描述
第一行输入两个正整数 N, M。表示 N 个文件之间拥有 M 条依赖关系。
后续 M 行，每行两个正整数 S 和 T，表示 T 文件依赖于 S 文件。

输出描述
输出共一行，如果能处理成功，则输出文件顺序，用空格隔开。
如果不能成功处理（相互依赖），则输出 -1。

输入示例
5 4
0 1
0 2
1 3
2 4
输出示例
0 1 2 3 4

提示信息
文件依赖关系如下：

所以，文件处理的顺序除了示例中的顺序，还存在
0 2 4 1 3
0 2 1 3 4
等等合法的顺序。

数据范围：
0 <= N <= 10 ^ 5
1 <= M <= 10 ^ 9
每行末尾无空格。
"""
from collections import deque, defaultdict


def topological_sort(n: int, edges: list) -> None:
    """
    拓扑排序算法
    :param n:   顶点的数量
    :param edges:   所有边的集合
    :return:    None
    """
    # 记录每个文件的入度
    in_degree = [0] * n
    # 用来记录每个文件的依赖关系
    umap = defaultdict(list)

    # 构建依赖图和入度表
    for u, v in edges:
        in_degree[v] += 1
        umap[u].append(v)

    # 初始化队列，先放入所有入度为0的节点
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    res = []

    while queue:
        cur = deque.popleft()
        res.append(cur)

        for sub_file in umap[cur]:
            in_degree[sub_file] -= 1
            if in_degree[sub_file] == 0:
                queue.append(sub_file)

    # 判断并且输出
    if len(res) == n:
        print(" ".join(map(str, res)))
    else:
        print(-1)


if __name__ == '__main__':
    n, m = map(int, input().split())

    edges = []
    for _ in range(m):
        u, v = map(int, input().split())
        edges.append([u, v])

    topological_sort(n, edges)
