# -*- coding: utf-8 -*
"""
leetcode 25 k个一组，翻转链表

给你链表的头节点 head ，每 k 个节点一组进行翻转，请你返回修改后的链表。

k 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。

你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。
"""
# Definition for singly-linked list.
class ListNode:
    """
    定义一个链表的节点class
    """
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse(head: ListNode, tail: ListNode) -> list[ListNode]:
    """
    将一个头指针为head，尾指针为tail的链表进行翻转
    :param head:    链表的头指针
    :param tail:    链表的尾指针
    :return:    返回翻转后的头尾指针
    """
    # 定义两个指针，分别是prev和p
    prev = tail.next
    p = head

    while prev != tail:
        # 进行翻转
        nex = p.next
        p.next = prev
        # 更新两个指针
        prev = p
        p = nex
    # 返回结果
    return tail, head


def reverseKGroup(head: ListNode, k: int) -> ListNode:
    """
    k个一组，翻转链表
    :param head:  需要进行翻转的链表
    :param k:   k数值，即多少个链表作为一组
    :return:    返回翻转后的链表的头节点
    """
    # 定义一个虚拟头节点
    dummy = ListNode(-1)
    dummy.next = head
    # pre, head作为两个指针
    pre = dummy

    # 开始遍历
    while head:
        tail = pre
        # 查看剩余部分是否大于等于k
        for i in range(k):
            tail = tail.next
            # 假如已经不够k个节点了，就直接返回结果，不再进行翻转
            if tail is None:
                return dummy.next

        # 此时，需要翻转头尾节点为head和tail的链表
        nex = tail.next
        head, tail = reverse(head, tail)
        # 然后，把翻转后的子链表，重新拼接回原链表中
        pre.next = head
        tail.next = nex
        pre = tail
        head = tail.next

    return dummy.next

if __name__ == '__main__':
    head = ListNode(1)
    head.next = ListNode(2)
    head.next.next = ListNode(3)
    head.next.next.next = ListNode(4)
    head.next.next.next.next = ListNode(5)

    head = reverseKGroup(head, 2)

    # 遍历进行打印
    while head:
        print(head.val, end=" ")
        head = head.next
