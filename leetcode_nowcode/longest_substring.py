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
    cur_s = {}
    left = 0

    # 开始遍历
    for right in range(len(s)):
        cur = s[right]

        # 当前字符，不在窗口内，则添加进来，且取最长的长度保存下来
        if cur not in cur_s.keys() or cur_s[cur] == 0:
            cur_s[cur] = 1
            res = max(res, right - left + 1)
        else:
            # 当前字符，存在窗口内，则不断左移指针
            while cur_s[cur] != 0:
                cur_left = s[left]
                cur_s[cur_left] -= 1
                left += 1
            # 此时，窗口已经吐出cur这个字符，重新添加当前字符
            cur_s[cur] += 1

    # 返回结果
    return res


if __name__ == '__main__':
    s = "abcabcbb"
    res = length_longest_substring(s)
    print(res)
