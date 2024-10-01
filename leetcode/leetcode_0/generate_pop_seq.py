# -*- coding: utf-8 -*


class Solution:
    """
    解决出栈的所有顺序的类
    """
    def __init__(self):
        self.res = []
        self.push_seq = ""

    def backtracking(self, stack: list, current_pop_sequence: list, next_push_index: int) -> None:
        """
        回溯方法
        :param stack:   一个栈，不断压入字符
        :param current_pop_sequence:    出栈的字符串序列
        :param next_push_index: 下一个压入栈的字符
        :return:    返回所有出栈顺序的可能
        """
        # 如果当前出栈序列的长度与入栈序列相同，加入结果
        if len(current_pop_sequence) == len(self.push_seq):
            self.res.append("".join(current_pop_sequence))
            return

        # 如果下一个压入的字符仍然存在，进行入栈操作
        if next_push_index < len(self.push_seq):
            # 入栈操作
            stack.append(self.push_seq[next_push_index])
            self.backtracking(stack, current_pop_sequence, next_push_index + 1)
            # 回溯，移除最后压入的元素
            stack.pop()

        # 如果栈不为空，进行出栈操作
        if stack:
            # 出栈操作
            current_pop_sequence.append(stack.pop())
            self.backtracking(stack, current_pop_sequence, next_push_index)
            # 回溯，移除最后出栈的元素
            stack.append(current_pop_sequence.pop())

    def main(self, push_sequence: str) -> list[str]:
        """
        主方法，处理数据，调用回溯方法，进行返回
        :param push_sequence:   输入的入栈顺序
        :return:    返回所有可能的出栈顺序
        """
        self.res = []
        self.push_seq = push_sequence

        # 初始化栈和当前出栈序列
        self.backtracking([], [], 0)
        return self.res


if __name__ == '__main__':
    # 定义入栈的顺序
    push_sequence = "ABCD"
    test = Solution()

    # 生成所有可能的出栈顺序
    pop_sequences = test.main(push_sequence)

    # 输出结果
    print("所有可能的出栈顺序:")
    for sequence in pop_sequences:
        print(sequence)
