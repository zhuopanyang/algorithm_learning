# -*- coding: utf-8 -*
"""
有n件物品和一个最多能背重量为w 的背包。第i件物品的重量是weight[i]，得到的价值是value[i] 。
每件物品只能用一次，求解将哪些物品装入背包里物品价值总和最大。

背包最大重量为 4
物品如下：
        重量  价值
物品 0    1   15
物品 1    3   20
物品 2    4   30
"""

def bag_0_1(n: int, bagweight: int, weights: list, values: list, optimize: bool=True) -> int:
    """
    0-1背包问题
    :param n: 物品的数量
    :param bagweight: 最大的背包容量
    :param weights: 物品的重量数组
    :param values: 物品的价值数组
    :return: 返回最大背包容量下，所能获得得最大的价值
    """

    # 判断使用优化的一维数组，还是使用未优化的二维数组做法
    if optimize:
        # 【一维数组的做法】
        # 初始化dp数组
        dp = [0] * (bagweight + 1)
        dp[0] = 0

        # 开始遍历
        for i in range(n):
            # for j in range(weights[i], bagweight + 1): # 完全背包问题，正向遍历背包容量，可以重复取物品（物品数量无限制）
            for j in range(bagweight, weights[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - weights[i]] + values[i])

        # 返回结果
        return dp[bagweight]
    else:
        # 【二维数组的做法】
        # 初始化dp数组
        dp = [[0] * (bagweight + 1) for _ in range(n)]
        for j in range(weights[0], bagweight + 1):
            dp[0][j] = values[0]

        # 开始遍历，先遍历物品，后遍历背包重量
        for i in range(1, n):
            for j in range(1, bagweight + 1):
                # 判断当前重量，是否可以放当前物品
                if j < weights[i]:
                    dp[i][j] = dp[i - 1][j]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weights[i]] + values[i])

        print(dp[n - 1][bagweight])


if __name__ == '__main__':
    # 初始化
    n, bagweight = 3, 4
    weights = [1, 3, 4]
    values = [15, 20, 30]
    optimize = True

    res = bag_0_1(n, bagweight, weights, values, optimize)
    print(res)
