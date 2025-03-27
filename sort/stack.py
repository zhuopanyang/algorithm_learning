# -*- coding: utf-8 -*
"""
实现一个栈的数据结构
"""

class Stack:
    """
    栈的数据结构
    """
    def __init__(self):
        self.stack = []  # 用于存储栈元素
        self.max_stack = []  # 用于存储最大值
        self.min_stack = []  # 用于存储最小值

    def is_empty(self):
        """检查栈是否为空"""
        return len(self.stack) == 0

    def push(self, item):
        """将元素压入栈"""
        self.stack.append(item)
        # 更新最大值栈
        if self.max_stack:
            self.max_stack.append(max(item, self.max_stack[-1]))
        else:
            self.max_stack.append(item)
        # 更新最小值栈
        if self.min_stack:
            self.min_stack.append(min(item, self.min_stack[-1]))
        else:
            self.min_stack.append(item)

    def pop(self):
        """弹出栈顶元素"""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        self.max_stack.pop()
        self.min_stack.pop()
        return self.stack.pop()

    def peek(self):
        """查看栈顶元素"""
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self.stack[-1]

    def get_max(self):
        """获取栈的最大值"""
        if self.is_empty():
            raise IndexError("get max from empty stack")
        return self.max_stack[-1]

    def get_min(self):
        """获取栈的最小值"""
        if self.is_empty():
            raise IndexError("get min from empty stack")
        return self.min_stack[-1]

    def __str__(self):
        """返回栈的字符串表示"""
        return str(self.stack)


# 测试代码
if __name__ == "__main__":
    stack = Stack()
    stack.push(3)
    stack.push(1)
    stack.push(4)
    stack.push(2)

    print("栈内容:", stack)
    print("栈顶元素:", stack.peek())
    print("最大值:", stack.get_max())
    print("最小值:", stack.get_min())

    stack.pop()
    print("弹出栈顶元素后，栈内容:", stack)
    print("栈顶元素:", stack.peek())
    print("最大值:", stack.get_max())
    print("最小值:", stack.get_min())
