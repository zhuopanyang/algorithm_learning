# -*- coding: utf-8 -*
"""
leetcode 3 无重复字符的最长子串

给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串的长度。
"""

def length_longest_substring(s: str) -> int:
    """
    寻找无重复字符的最长子串的长度
    :param s:   输入的字符串
    :return:    返回子串的长度
    """
    res = 0
    s_length = len(s)

    # 输入边界处理
    if s_length < 1:
        return res

    # 定义指针、滑动窗口
    left, right = 0, 0
    match = 0
    windows = {} # 维持的滑动窗口

    # 开始遍历
    while right < s_length:
        c1 = s[right]
        right += 1

        # 判断当前字符，不在之前的滑动窗口内
        if c1 not in windows.keys() or windows.get(c1) < 1:
            # 更新滑动窗口，以及最长的子串大小
            windows[c1] = 1
            res = max(res, right - left)
        else:
            # 当前字符，已经存在之前的滑动窗口内
            windows[c1] = windows.get(c1) + 1
            match += 1

        # 检测到当前的滑动窗口内，存在两个相同的字符，需要移动left，知道找到为止
        while match == 1:
            c2 = s[left]
            left += 1

            windows[c2] = windows.get(c2) - 1
            if windows[c2] == 1:
                match -= 1

    # 返回结果
    return res


if __name__ == '__main__':
    s = "abcabcbb"
    res = length_longest_substring(s)
    print(res)
