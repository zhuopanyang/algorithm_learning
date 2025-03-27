# -*- coding: utf-8 -*
"""
有m个数组，每个数组都有n个非负正整数，我们从每个数组中选出一个数字
组成一个序列，可以组成n*m个序列，

请给出序列和最小的k个数值
"""
import heapq


def find_k_smallest_sums(arrays, k):
    if not arrays or k <= 0:
        return []

    # 初始化最小堆
    heap = []
    m = len(arrays)
    n = len(arrays[0])

    # 创建一个数组来记录每个数组的当前索引
    indices = [0] * m

    # 计算初始和（所有数组的第一个元素）
    current_sum = sum(arrays[i][0] for i in range(m))
    heapq.heappush(heap, (current_sum, indices.copy()))

    # 使用一个集合来避免重复的索引组合
    visited = set()
    visited.add(tuple(indices))

    result = []

    while len(result) < k and heap:
        current_sum, indices = heapq.heappop(heap)
        result.append(current_sum)

        # 生成下一个可能的索引组合
        for i in range(m):
            if indices[i] + 1 < n:
                new_indices = indices.copy()
                new_indices[i] += 1
                new_sum = current_sum - arrays[i][indices[i]] + arrays[i][new_indices[i]]

                if tuple(new_indices) not in visited:
                    heapq.heappush(heap, (new_sum, new_indices))
                    visited.add(tuple(new_indices))

    return result


# 示例用法
if __name__ == "__main__":
    # 示例输入
    arrays = [
        [1, 3, 5],
        [2, 4, 6],
        [7, 8, 9]
    ]
    k = 5
    result = find_k_smallest_sums(arrays, k)
    print("最小的k个序列和是：", result)
