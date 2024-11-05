# -*- coding: utf-8 -*

direction = [
    [0, 1],
    [1, 0],
    [0, -1],
    [-1, 0]
]


def dfs(grid, visited, x, y):
    # 终止条件, 遇到访问过的节点，或者遇到海洋的节点，就退出
    if visited[x][y] or grid[x][y] == 0:
        return
    # 修改当前节点为访问过
    visited[x][y] = True

    for i, j in direction:
        next_x = x + i
        next_y = y + j

        # 检查一下，下标是否越界，如果是的话，直接跳过
        if next_x < 0 or next_x >= len(grid) or next_y < 0 or next_y >= len(grid[0]):
            continue

        # 不断深度遍历
        dfs(grid, visited, next_x, next_y)


if __name__ == "__main__":
    n, m = 4, 5

    grid = [
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1],
    ]
    # for i in range(n):
    #     row = list(map(int, input().split()))
    #     grid.append(row)

    # 定义一个访问表
    visited = [[False] * m for _ in range(n)]

    res = 0
    for i in range(n):
        for j in range(m):
            # 判断，如果当前节点是陆地，res+1, 并尽可能地深度搜索相邻的；陆地
            if grid[i][j] == 1 and not visited[i][j]:
                res += 1
                dfs(grid, visited, i, j)

    print(res)
