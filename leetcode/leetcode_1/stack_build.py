# -*- coding: utf-8 -*
"""
自己设置最大堆、最小堆的数据结构
"""

class MinHeap:
    """
    定义一个最小堆的数据结构
    """
    def __init__(self):
        self.heap = []

    def insert(self, item: int):
        """
        插入元素到堆中
        :param item:    插入的元素
        :return:    None
        """
        self.heap.append(item)
        self._sift_up(len(self.heap) - 1)

    def extract_min(self):
        """
        提取堆顶最小元素
        :return:    弹出返回堆顶的最小元素
        """
        # 边界处理
        if len(self.heap) == 0:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()

        # 将最后一个元素与堆顶交换
        min_item = self.heap[0]
        self.heap[0] = self.heap.pop()
        # 将该元素进行下沉
        self._sift_down(0)
        return min_item

    def peek(self):
        """
        查看堆顶元素
        :return: 返回堆顶的最小元素
        """
        return self.heap[0] if self.heap else None

    def size(self):
        """
        获取堆的大小
        :return:  返回堆的大小
        """
        return len(self.heap)

    def _sift_up(self, index: int):
        """
        向上调整（用于插入元素后）
        （1）比较当前元素与它的父节点。
        （2）如果当前元素的值小于父节点的值，则交换它们的位置。
        （3）继续向上比较，直到当前元素的值大于或等于父节点的值，或者到达堆顶。
        :param index:   插入的元素的位置
        :return:    None
        """
        parent = (index - 1) // 2
        while index > 0 and self.heap[index] < self.heap[parent]:
            self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
            index = parent
            parent = (index - 1) // 2

    def _sift_down(self, index):
        """
        向下调整（用于提取最小元素后）
        （1）比较当前节点与它的左子节点和右子节点。
        （2）选择子节点中较小的那个，如果当前节点的值大于这个较小的子节点，则交换它们的位置。
        （3）继续向下比较，直到当前节点的值不大于子节点的值，或者到达叶子节点。
        :param index:   提取后的元素的位置
        :return:    None
        """
        left_child = 2 * index + 1
        right_child = 2 * index + 2
        smallest = index

        # 当前节点和左子节点进行比较
        if left_child < len(self.heap) and self.heap[left_child] < self.heap[smallest]:
            smallest = left_child

        # 当前节点和右子节点进行比较
        if right_child < len(self.heap) and self.heap[right_child] < self.heap[smallest]:
            smallest = right_child

        # 将三者中的最小值，跟当前节点，进行交换
        if smallest != index:
            self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
            # 交换之后，继续不断下沉
            self._sift_down(smallest)

    def build_heap(self, nums: list):
        """
        从列表构建堆
        :param nums: 给定的列表
        :return:    None
        """
        self.heap = nums[:]
        for i in range(len(self.heap) // 2 - 1, -1, -1):
            self._sift_down(i)


# 示例
if __name__ == "__main__":
    heap = MinHeap()

    # 插入元素
    heap.insert(3)
    heap.insert(1)
    heap.insert(4)
    heap.insert(0)
    heap.insert(5)

    # 查看堆顶元素
    print("堆顶元素:", heap.peek())  # 输出: 0

    # 提取最小元素
    min_element = heap.extract_min()
    print("提取的最小元素:", min_element)  # 输出: 0
    print("堆顶元素:", heap.peek())  # 输出: 1

    # 构建一个堆
    lst = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    heap.build_heap(lst)
    print("堆顶元素:", heap.peek())  # 输出: 1

    # 打印最小堆
    print(heap.heap)

    # 提取所有元素
    while heap.size() > 0:
        print(heap.extract_min(), end=" ")  # 输出: 1 2 3 4 5 6 7 8 9
