# 删除排序数组中的重复项
class Solution26:
    def removeDuplicates(self, nums: List[int]) -> int:
        if len(nums) <= 1:
            return len(nums)
        j = 0
        for i in range(1, len(nums)):
            if nums[i] != nums[j]:
                j += 1
                nums[j] = nums[i]
        return j + 1

# 旋转数组
class Solution189:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        k = k % len(nums)
        if len(nums) > 1 and k != 0:
            nums[:] = nums[-k:] + nums[:len(nums) - k]

# 合并两个有序数组
class Solution88:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        p1 = m - 1
        p2 = n - 1
        for i in range(n + m - 1, -1, -1):
            if p2 < 0:
                break
            if p1 >= 0 and nums1[p1] > nums2[p2]:
                nums1[i] = nums1[p1]
                p1 -= 1
            else:
                nums1[i] = nums2[p2]
                p2 -= 1

# 合并两个有序链表
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution21:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1:
            return l2
        if not l2:
            return l1
        if l1.val <= l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2

# 两数之和
class Solution1:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        cache = {}
        for i in range(len(nums)):
            complement = target - nums[i]
            if complement in cache:
                return [cache[complement], i]
            cache[nums[i]] = i

# 移动零
class Solution283:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        left = right = 0
        while right < len(nums):
            if nums[right] != 0:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
            right += 1

# 加一
class Solution66:
    def plusOne(self, digits: List[int]) -> List[int]:
        for i in range(len(digits) - 1, -1, -1):
            digits[i] = (digits[i] + 1) % 10
            if digits[i] != 0:
                return digits
        if digits[0] == 0:
            return [1] + digits

# 641:设计循环双端队列
class MyCircularDeque:

    def __init__(self, k: int):
        """
        Initialize your data structure here. Set the size of the deque to be k.
        """
        self.size = k
        self.q = []

    def insertFront(self, value: int) -> bool:
        """
        Adds an item at the front of Deque. Return true if the operation is successful.
        """
        if len(self.q) < self.size:
            self.q = [value] + self.q
            return True
        return False

    def insertLast(self, value: int) -> bool:
        """
        Adds an item at the rear of Deque. Return true if the operation is successful.
        """
        if len(self.q) < self.size:
            self.q.append(value)
            return True
        return False

    def deleteFront(self) -> bool:
        """
        Deletes an item from the front of Deque. Return true if the operation is successful.
        """
        if len(self.q) == 0:
            return False
        self.q.pop(0)
        return True

    def deleteLast(self) -> bool:
        """
        Deletes an item from the rear of Deque. Return true if the operation is successful.
        """
        if len(self.q) == 0:
            return False
        self.q.pop()
        return True

    def getFront(self) -> int:
        """
        Get the front item from the deque.
        """
        if len(self.q) == 0:
            return -1
        return self.q[0]

    def getRear(self) -> int:
        """
        Get the last item from the deque.
        """
        if len(self.q) == 0:
            return -1
        return self.q[-1]

    def isEmpty(self) -> bool:
        """
        Checks whether the circular deque is empty or not.
        """
        return len(self.q) == 0

    def isFull(self) -> bool:
        """
        Checks whether the circular deque is full or not.
        """
        return len(self.q) == self.size




# 接雨水
class Solution42:
    #DP
    def trap(self, height: List[int]) -> int:
        if len(height) == 0:
            return 0
        ans = 0
        left = [0] * len(height)
        right = [0] * len(height)
        left[0] = height[0]
        for i in range(1, len(height)):
            left[i] = max(height[i], left[i - 1])
        right[-1] = height[-1]
        for i in range(len(height)-2, -1, -1):
            right[i] = max(height[i], right[i + 1])
        for i in range(len(height)):
            ans += min(left[i], right[i]) - height[i]
        return ans

    # 2 pointer
    def trap(self, height: List[int]) -> int:
        if len(height) == 0:
            return 0
        left, right = 0, len(height) - 1
        ans, left_max, right_max = 0, 0, 0
        while left < right:
            if height[left] <= height[right]:
                left_max = max(left_max, height[left])
                ans += left_max - height[left]
                left += 1
            else:
                right_max = max(right_max, height[right])
                ans += right_max - height[right]
                right -= 1
        return ans


#=======================================================================================================
#下周预习

# 有效的字母异位词
class Solution242:
    def isAnagram(self, s: str, t: str) -> bool:
        return sorted(s) == sorted(t)

# 二叉树中序遍历
class Solution94:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        stack = []
        cur = root
        while cur or len(stack) != 0:
            while cur:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            res.append(cur.val)
            cur = cur.right
        return res

# 最小的k个数
class Solution剑指40:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        return sorted(arr)[:k]