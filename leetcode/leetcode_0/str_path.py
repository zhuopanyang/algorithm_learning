# -*- coding:utf-8 -*
"""
字典 strList 中从字符串 beginStr 和 endStr 的转换序列是一个按下述规格形成的序列：
1. 序列中第一个字符串是 beginStr。
2. 序列中最后一个字符串是 endStr。
3. 每次转换只能改变一个字符。
4. 转换过程中的中间字符串必须是字典 strList 中的字符串，且strList里的每个字符串只用使用一次。
给你两个字符串 beginStr 和 endStr 和一个字典 strList，找到从 beginStr 到 endStr 的最短转换序列中的字符串数目。
如果不存在这样的转换序列，返回 0。

第一行包含一个整数 N，表示字典 strList 中的字符串数量。
第二行包含两个字符串，用空格隔开，分别代表 beginStr 和 endStr。
后续 N 行，每行一个字符串，代表 strList 中的字符串。

输出一个整数，代表从 beginStr 转换到 endStr 需要的最短转换序列中的字符串数量。如果不存在这样的转换序列，则输出 0。

"""


def judge(s1, s2):
    """
    判断当前两个字符串之间是否只有距离1
    :param s1:  第一个字符串 s1
    :param s2:  第二个字符串 s2
    :return:    返回二者之间的距离相差1
    """
    count = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            count += 1
    return count == 1

def bfs(n: int, begin_str: str, end_str: str, str_list: list):
    """
    广度优先遍历算法
    :param n:   字典str_list中字符串的数量
    :param begin_str:   初始的字符串
    :param end_str:     结尾的字符串
    :param str_list:    字典
    :return:
    """
    # 使用bfs
    visited = [False for _ in range(n)]
    queue = [[begin_str, 1]]
    while queue:
        s, step = queue.pop(0)

        # 判断是否到终点了，这样子只能获得其中一种结果
        # 需要用回溯算法，才能获得所有可能的结果
        if judge(s, end_str):
            print(step + 1)
            exit()

        # 开始遍历，走下一步
        for i in range(n):
            if visited[i] == False and judge(str_list[i], s):
                visited[i] = True
                queue.append([str_list[i], step + 1])

    # 都没找到最终点的一条路线
    print(0)

if __name__ == "__main__":
    n = int(input())
    begin_str, end_str = map(str, input().split())
    if begin_str == end_str:
        print(0)
        exit()

    str_list = []
    for i in range(n):
        str_list.append(input())

    # 使用广度优先遍历算法
    bfs(n, begin_str, end_str, str_list)
