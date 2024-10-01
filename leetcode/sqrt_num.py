# -*- coding: utf-8 -*


def sqrt_binary_search(target: int, precision: float) -> float:
    """
    计算一个数字的根号，保留两位小数点
    :param target:  需要计算的数字
    :param precision:   需要保留的小数点
    :return:    返回保留小数点后的数字的根号
    """
    left = 0
    right = target
    mid = 0

    while right - left > precision:  # 保持精确度，用 []
        mid = (left + right) / 2
        mid_squared = mid * mid

        if mid_squared < target:  # mid 的平方小于目标
            left = mid
        else:  # mid 的平方大于等于目标
            right = mid

    return round(mid, 2)  # 保留两位小数


if __name__ == '__main__':
    # 要求解根号的数字
    number = 10
    # 精确度到小数点后两位
    precision = 0.01

    result = sqrt_binary_search(number, precision)
    print(f"根号{number}为：{result}")
