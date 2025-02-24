# -*- coding: utf-8 -*
"""
【A*算法】
127. 骑士的攻击
题目描述
在象棋中，马和象的移动规则分别是“马走日”和“象走田”。
现给定骑士的起始坐标和目标坐标，要求根据骑士的移动规则，计算从起点到达目标点所需的最短步数。
棋盘大小 1000 x 1000（棋盘的 x 和 y 坐标均在 [1, 1000] 区间内，包含边界）

输入描述
第一行包含一个整数 n，表示测试用例的数量，1 <= n <= 100。
接下来的 n 行，每行包含四个整数 a1, a2, b1, b2，分别表示骑士的起始位置 (a1, a2) 和目标位置 (b1, b2)。

输出描述
输出共 n 行，每行输出一个整数，表示骑士从起点到目标点的最短路径长度。

输入示例
6
5 2 5 4
1 1 2 2
1 1 8 8
1 1 8 7
2 1 3 3
4 6 4 6

输出示例
2
4
6
5
1
0
"""

import heapq


def distance(a: list, b: list) -> int:
    """
    计算两个点之间的欧拉距离
    :param a:   第一个节点
    :param b:   第二个节点
    :return:    返回两个节点之间的欧拉距离
    """
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def bfs(start: list, end: list, moves: list) -> int:
    """
    深度优先遍历算法 + A*算法（在于利用启发式规则 + 最小堆）
    :param start:   起点
    :param end: 终点
    :param moves:   骑士可以走的方向
    :return:    返回骑士从起点到终点的最短路径长度
    """
    # 定义一个最小堆，存放（distance, start）
    # 其中distance = 起点到cur的距离 + cur到终点的距离（初始cur为start起点）
    q = [(distance(start, end), start)]
    # 用来记录，从起点到cur当前节点的最短距离
    step = {start: 0}

    # 遍历最小堆是否为空
    while q:
        # 【A*算法的启发式规则（可自定义）】：取出堆中，最小距离的一个点
        d, cur = heapq.heappop(q)

        # 终止条件，遇到终点
        if cur == end:
            return step[cur]

        # 开始遍历，骑士下一步的可能性
        for move in moves:
            next_node = (move[0] + cur[0], move[1] + cur[1])

            # 判断下一步节点，是否可以移动
            if next_node[0] < 1 or next_node[1] < 1 or next_node[0] > 1000 or next_node[1] > 1000:
                continue

            # 发现新的最短距离，进行更新，并且把该节点next_node，放入到最小堆中
            step_new = step[cur] + 1
            if step_new < step.get(next_node, float('inf')):
                step[next_node] = step_new
                # step_new + distance(next_node, end)表示 起点到next_node的距离 + next_node到终点的距离
                heapq.heappush(q, (step_new + distance(next_node, end), next_node))

    # 没找到，则返回-1
    return -1


if __name__ == '__main__':
    n = int(input())
    # 定义骑士的下一步的规则
    moves = [(1, 2), (2, 1), (-1, 2), (2, -1), (1, -2), (-2, 1), (-1, -2), (-2, -1)]

    for _ in range(n):
        a1, a2, b1, b2 = map(int, input().split())
        print(bfs((a1, a2), (b1, b2)))
