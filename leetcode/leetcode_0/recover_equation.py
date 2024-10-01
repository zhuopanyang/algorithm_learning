# -*- coding: utf-8 -*
"""
1）题目描述：
用python写代码解答，给出一些仅包含正整数、加号、乘号和等号的方程，请判断这些方程能否通过插入至多一个数位（若原方程成立则可以不插）
使得方程成立。插入一个数位即将方程视为一个字符串，并将一个0到9之间的数插入中间，开头或末尾。
2）输入描述：
第一行有一个正整数T（1<=T<=10），代表方程的数量。
接下来T行，每行均有一个仅包含十进制正整数，加号和乘号的方程。每个方程中均只会包含一个等号。保证输入的方程合法，即每个数均不含前导零，
开头和末尾没有运算符，且没有两个相邻的运算符。输入中方程两边计算结果的最大值不超过1000000000，且每个方程的长度不超过1000。
3）输出描述：
对于每个方程，若其成立或可以通过往该方程中插入一个数位使得方程成立，则输出Yes，否则输出No。
4）样例输入如下：
6
16=1+2*3
7*8*9=54
1+1=1+22
4*6=22+2
15+7=1+2
11+1=1+5
"""
import sys


def eval_expression(exp: str) -> int:
    """
    计算简单数学表达式的值，只包含正整数、加号和乘号
    :param exp: 数学表达式的字符串形式
    :return:    返回数学表达式的计算返回值
    """
    try:
        return eval(exp)
    except:
        return None


def can_insert_digit_to_make_valid(equation: str) -> str:
    """
    评估左侧和右侧的表达式，是否成立，或者插入一个数位，能否成立
    :param equation:   输入的数学表达式
    :return: 返回Yes或者No
    """

    left, right = equation.split('=')

    # 如果原方程已经成立，直接返回Yes
    if eval_expression(left) == eval_expression(right):
        return "Yes"

    # 尝试在左侧表达式或右侧数字中插入0到9的任意一个数字
    for i in range(10):
        digit = str(i)

        # 试图在右侧数字中插入
        for j in range(len(right) + 1):
            new_right = right[:j] + digit + right[j:]
            if eval_expression(left) == eval_expression(new_right):
                return "Yes"

        # 试图在左侧表达式中插入
        for k in range(len(left) + 1):
            new_left = left[:k] + digit + left[k:]
            if eval_expression(new_left) == eval_expression(right):
                return "Yes"

    return "No"


# 主程序，读入输入并输出结果
if __name__ == "__main__":
    # 读取输入
    T = 6
    equations = [
        "16=1+2*3",
        "7*8*9=54",
        "1+1=1+22",
        "4*6=22+2",
        "15+7=1+2",
        "11+1=1+5"
    ]

    # 处理每个方程并输出结果
    for eq in equations:
        result = can_insert_digit_to_make_valid(eq)
        print(result)
