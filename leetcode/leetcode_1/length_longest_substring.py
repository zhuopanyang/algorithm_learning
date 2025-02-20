# -*- coding: utf-8 -*
"""
Leetcode编程题3. 无重复字符的最长子串

给定一个字符串 s ，请你找出其中不含有重复字符的最长子串的长度。

输入: s = "abcabcbb"
输出: 3
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
【滑动窗口】
"""

def lengthOfLongestSubstring(s: str) -> int:
    """
    返回一个字符串中，不含有重复字符的最长子串的长度
    :param s: 输入的字符串s
    :return: 返回的最长子串的长度
    """
    # res用于记录最长的长度
    res = 0
    # window用来记录一个滑动窗口
    window = {}
    # left，right分别表示滑动窗口的左右指针
    left = 0

    # 开始遍历整个字符串
    for right in range(len(s)):
        # 取出当前的字符
        cur = s[right]

        # 当前字符，不在窗口内，则添加进来，且取最长的长度保存下来
        if cur not in window.keys() or window[cur] == 0:
            window[cur] = 1
            res = max(res, right - left + 1)
        else:
            # 当前字符，存在窗口内，则不断左移指针
            while window[cur] != 0:
                cur_left = s[left]
                window[cur_left] -= 1
                left += 1
            # 此时，窗口已经吐出cur这个字符，重新添加当前字符
            window[cur] += 1

    # 返回结果
    return res

if __name__ == '__main__':
    res = lengthOfLongestSubstring("abcabcbb")
    print(res)
