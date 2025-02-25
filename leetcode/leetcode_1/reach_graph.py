# -*- coding: utf-8 -*
"""
105. 有向图的完全可达性
题目描述
给定一个有向图，包含 N 个节点，节点编号分别为 1，2，...，N。现从 1 号节点开始，
如果可以从 1 号节点的边可以到达任何节点，则输出 1，否则输出 -1。

输入描述
第一行包含两个正整数，表示节点数量 N 和边的数量 K。 后续 K 行，每行两个正整数 s 和 t，表示从 s 节点有一条边单向连接到 t 节点。

输出描述
如果可以从 1 号节点的边可以到达任何节点，则输出 1，否则输出 -1。

输入示例
4 4
1 2
2 1
1 3
2 4

输出示例
1
"""
import collections


def bfs(root: int, graph: list) -> set:
    """
    广度优先遍历法
    :param root:    输入的第一个节点的位置
    :param graph:   输入的图
    :return:    返回从第一个节点，所有的可达到节点的set集合
    """
    res = set()
    queue = collections.deque([root])

    while queue:
        cur = queue.popleft()
        res.add(cur)

        # 遍历所有当前节点的所有可达节点
        for neighbor in graph[cur]:
            queue.append(neighbor)
        # 将刚遍历过的节点的数据清空
        graph[cur] = []

    return res


def dfs(graph: list, key: int, visited: list) -> None:
    """
    深度优先遍历方法
    :param graph:   输入的图
    :param key: 输入的当前的节点
    :param visited: 记录节点是否访问过
    :return:    None
    """
    for neighbor in graph[key]:
        if visited[neighbor] == False:
            visited[neighbor] = True
            dfs(graph, neighbor, visited)


def main(solution_type="DFS") -> None:
    """
    主函数
    :return:
    """
    N, K = map(int, input().split())
    graph = [[] for _ in range(K + 1)]

    for _ in range(K):
        s, t = map(int, input().split())
        graph[s].append(t)

    visited = [False for _ in range(N + 1)]

    # 初始化第一节点，开始被遍历
    visited[1] = True

    if solution_type == "DFS":
        # （1）dfs
        dfs(graph, 1, visited)
        for i in range(1, N + 1):
            if not visited[i]:
                print(-1)
                return
        # 全部都能访问到
        print(1)
    else:
        # （2）bfs
        res = bfs(1, graph)
        if len(res) == N:
            print(1)
        else:
            print(-1)


if __name__ == "__main__":
    # 选择 BFS 还是 DFS算法
    solution_type = "DFS"
    main(solution_type)
