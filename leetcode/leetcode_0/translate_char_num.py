# -*- coding: utf-8 -*
"""
用python解答，"1"表示"a"，"2"表示"b"，"3"表示"c"，以此类推，"26"表示"z"，
给定一个输入的数字字符串"11231012"，其中可以组成一个数字，也可以组成两个数字，来翻译成字母字符，
请给出该数字字符串的所有可能的翻译结果。

【回溯算法】
"""


class Solution:
    """
    整体的解决方法类
    """

    def __init__(self):
        self.res = set()    # 使用set()来去重
        self.path = ""  # 当前的翻译结果
        self.number_string = ""     # 输入的数字字符串

    def backtracking(self, start_index: int) -> None:
        """
        回溯法
        :param start_index: 当前数字的索引
        :return: 返回翻译结果
        """
        # 如果遍历到字符串末尾，将路径加入结果集
        if start_index == len(self.number_string):
            self.res.add(self.path)
            return

        # 尝试单个数字
        num = int(self.number_string[start_index])
        if 1 <= num <= 9:  # 1到9是有效的字符范围
            self.path += chr(ord('a') + num - 1)
            self.backtracking(start_index + 1)
            self.path = self.path[:-1]

        # 尝试两个数字组成一个字符
        if start_index + 1 < len(self.number_string):
            num = int(self.number_string[start_index:start_index + 2])
            if 10 <= num <= 26:  # 10到26是有效的字符范围
                self.path += chr(ord('a') + num - 1)
                self.backtracking(start_index + 2)
                self.path = self.path[:-2]

    def dfs(self, start_index: int) -> None:
        """
        递归法
        :param start_index: 当前数字的索引
        :return: 返回的翻译结果
        """
        if start_index == len(self.number_string):
            # 找到一个完整的翻译结果，加入结果集
            self.res.add(self.path)
            return

        # 翻译一位数
        if start_index < len(self.number_string):
            num = int(self.number_string[start_index])
            if 1 <= num <= 9:
                self.path += chr(ord('a') + num - 1)
                self.dfs(start_index + 1)
                self.path = self.path[:-1]

        # 翻译两位数
        if start_index + 1 < len(self.number_string):
            num = int(self.number_string[start_index:start_index + 2])
            if 10 <= num <= 26:
                self.path += chr(ord('a') + num - 1)
                self.dfs(start_index + 2)
                self.path = self.path[:-2]

    def main(self, number_string: str) -> str:
        """
        主函数
        :param number_string: 输入的数字字符串
        :return:    返回的翻译结果
        """
        self.res = set()
        self.path = ""
        self.number_string = number_string

        # self.dfs(0)   # 递归方法求解
        self.backtracking(0)    # 回溯法求解

        return self.res


if __name__ == '__main__':
    # 示例输入
    number_string = "11231012"

    test = Solution()
    translated_strings = test.main(number_string)

    print(translated_strings)
