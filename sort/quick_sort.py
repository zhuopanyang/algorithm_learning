# -*- coding: utf-8 -*
"""
快速排序的实现
"""


def quick_sort_inplace(arr: list, low: int, high: int) -> None:
    """
    快速排序方法
    :param arr: 输入的数组
    :param low: 数组的左边界
    :param high:    数组的右边界
    :return:    None
    """
    # 开始分治法
    if low < high:
        pi = partition(arr, low, high)
        quick_sort_inplace(arr, low, pi - 1)
        quick_sort_inplace(arr, pi + 1, high)


def partition(arr: list, low: int, high: int) -> int:
    pivot = arr[high]  # 选择最后一个元素作为基准
    # 将数组排序成，从low到最后的i处的元素都 < pivot，从最后的i+1到high处的元素都 > pivot
    i = low - 1

    # 开始遍历
    for j in range(low, high):
        # 遇到比基准低的元素，则进行互换元素
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    # 遍历结束后，基准元素和i + 1的位置元素，进行互换
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


if __name__ == '__main__':
    # 示例数组
    arr = [10, 7, 8, 9, 1, 5]
    quick_sort_inplace(arr, 0, len(arr) - 1)
    print(arr)
