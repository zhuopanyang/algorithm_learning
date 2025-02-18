# -*- coding: utf-8 -*
"""

"""


def test():
    s = "abcbcad"

    res = 1
    cur_s = map()
    left = 0

    # 开始遍历
    for i in range(len(s)):
        cur = s[i]

        # 先判断当前字符，在不在窗口内，还有个数量
        if cur not in cur_s.keys():
            cur_s[cur] = 1

            tmp = 0
            for key, value in enumerate(cur_s):
                tmp += value

            res = max(res, tmp)

        if cur in cur_s.keys() and cur_s[cur] > 1:
            # 不断左移指针
            while cur_s[cur] != 1:
                cur_left = cur_s[left]
                cur_s[cur_left] -= 1
                left += 1

    print(res)

def main():
    test()


if __name__ == "__main__":
    main()
