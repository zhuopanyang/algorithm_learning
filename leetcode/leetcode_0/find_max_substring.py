# -*- coding: utf-8 -*
"""
有一个全是小写字母的字符串，统计一下长度大于等于k的连续字符子串的数量，并输出数量最多的子串

1) 遍历字符串，使用一个字典来记录所有符合长度要求的子串及其出现次数。
2) 在每次迭代中，如果当前字符与前一个字符相同，则增加当前子串的长度；如果不同，则重置子串长度并更新字典。
3) 在迭代结束后，找出字典中数量最多的子串。
"""


def count_substrings(s, k):
    if k <= 0 or not s:
        return 0, ""

    substring_count = {}
    n = len(s)

    # 连续字符子串的子串
    for i in range(n):
        for length in range(k, n - i + 1):  # 长度从 k 到最大可能的长度
            substring = s[i:i + length]  # 获取当前子串
            if substring in substring_count:
                substring_count[substring] += 1
            else:
                substring_count[substring] = 1

    # 找到出现次数最多的子串
    max_count = 0
    most_frequent_substring = ""

    for substring, count in substring_count.items():
        if count > max_count:
            max_count = count
            most_frequent_substring = substring

    return sum(1 for count in substring_count.values() if count > 0), most_frequent_substring


# 示例用法
s = "aabbccddeeffggg"
k = 3
count, frequent_substring = count_substrings(s, k)
print(f"长度大于等于 {k} 的连续字符子串的数量: {count}")
print(f"出现次数最多的子串: '{frequent_substring}'")
