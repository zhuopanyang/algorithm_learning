# -*- coding: utf-8 -*
"""
打印一个等边三角形

例如：
size = 4
...*        （4 - 1 - 1）=  2 个. + 1个.*    第一层
..*.*       （4 - 2 - 1）=  1 个. + 2个.*    第二层
.*.*.*      （4 - 3 - 1）=  0 个. + 3个.*    第三层
*.*.*.*     （4 - 4 - 1）= -1 个. + 4个.*    第四层

"""

def print_eq_triangle(size: int):
    """
    打印一个等边三角形
    :param size: 等边三角形的边大小
    :return:    None
    """
    # 开始遍历
    for i in range(1, size + 1):

        tmp = size - i - 1
        if tmp >= 0:
            # 先打印前缀的.
            print("." * tmp, end="")
            print(".*" * i, end="")
        else:
            print("*", end="")
            print(".*" * (i - 1), end="")

        # 每一行进行换行
        print("")


if __name__ == '__main__':
    print_eq_triangle(4)
