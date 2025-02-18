# -*- coding: utf-8 -*
"""

"""


def test():
    s = "tmmzuxt"

    res = 0
    cur_s = {}
    left = 0

    # 开始遍历
    for i in range(len(s)):
        cur = s[i]

        # 当前字符，不在窗口内，则添加进来，且取最长的长度保存下来
        if cur not in cur_s.keys() or cur_s[cur] == 0:
            cur_s[cur] = 1
            res = max(res, sum(cur_s.values()))
        else:
            # 当前字符，存在窗口内，则不断左移指针
            while cur_s[cur] != 0:
                cur_left = s[left]
                cur_s[cur_left] -= 1
                left += 1
            # 此时，窗口已经吐出cur这个字符，重新添加当前字符
            cur_s[cur] += 1

    # 返回结果
    print(res)

def main():
    test()


if __name__ == "__main__":
    main()
