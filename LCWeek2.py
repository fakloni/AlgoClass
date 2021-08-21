'''
HashMap小总结：
HashMap本质就是Key-value Pair，然后能达到读取都在O(1)的数据结构
理想状况下，每一个key都有一个unique hash value，所以哈希函数最好能把输入键值都分布在一个相较疏散的区域从而保证
当发生哈希碰撞的时候，移动的元素也不会很多。
'''


# 242 有效的字母异位词
# python的sorted()是O(nlogn) 也就是这个的时间复杂度
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        return sorted(s) == sorted(t)

# 用哈希表完成，时间复杂度是O(n)，空间复杂度是O(1)
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        letters = [0] * 26
        for i in s:
            letters[ord(i) - ord('a')] += 1
        for j in t:
            letters[ord(j) - ord('a')] -= 1
        return all([i == 0 for i in letters])

# 1 两数之和
# 哈希表方法，时间复杂度是O(n)，空间复杂度是O(n)
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        cache = {}
        for i in range(len(nums)):
            complement = target - nums[i]
            if complement in cache:
                return [cache[complement], i]
            cache[nums[i]] = i


# 589 N叉树的前序遍历
# 递归方法 DFS
class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        res = []
        def dfs(root):
            if root:
                res.append(root.val)
                for child in root.children:
                    dfs(child)
        dfs(root)
        return res
# 迭代，用一个stack来控制节点，时间复杂度和空间复杂度为O(M)，M为节点的个数
class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        if not root:
            return []
        stack, res = [root], []
        while stack:
            node = stack.pop()
            res.append(node.val)
            stack.extend(node.children[::-1])
        return res

# HeapSort ：自学  https://www.geeksforgeeks.org/heap-sort/

# 49 字母异位词分组
# 用字母出现频率的tuple作为dict的key，因为元组是immutable，也可以用sorted()来做
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        res = collections.defaultdict(list)
        for word in strs:
            letter = [0] * 26
            for i in word:
                letter[ord(i) - ord('a')] += 1
            res[tuple(letter)].append(word)
        return res.values()


# 94 二叉树的中序遍历
# 迭代和递归时间空间复杂度都为O(n), n: number of nodes
# 迭代方法
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res, stack = [], []
        while root or len(stack) != 0:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            res.append(root.val)
            root = root.right
        return res
# 递归方法
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        def dfs(root):
            if root:
                dfs(root.left)
                res.append(root.val)
                dfs(root.right)
        res = []
        dfs(root)
        return res


# 144 二叉树的前序遍历
# 迭代方法
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        res, stack = [], [root]
        while stack:
            cur = stack.pop()
            res.append(cur.val)
            if cur.right:
                stack.append(cur.right)
            if cur.left:
                stack.append(cur.left)
        return res
# 递归方法
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        def preorder(root):
            if root:
                res.append(root.val)
                preorder(root.left)
                preorder(root.right)
        preorder(root)
        return res


# 429 N叉树的层序遍历
# O(n)的迭代方法，用双端队列
class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        if not root:
            return []
        res = []
        q = collections.deque([root])
        while q:
            level = []
            for i in range(len(q)):
                node = q.popleft()
                level.append(node.val)
                q.extend(node.children)
            res.append(level)
        return res

# 递归方法
class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:

        def traverse(root, level):
            if len(res) == level:
                res.append([])
            res[level].append(root.val)
            for child in root.children:
                traverse(child, level + 1)
        res = []
        if root:
            traverse(root, 0)
        return res

# 丑数（字节跳动在半年内面试中考过）
# 用python的小顶堆来做，时间复杂度为O(nlogn)
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        heap = [1]
        seen = {1}
        factor = [2, 3, 5]
        for i in range(n-1):
            curr = heapq.heappop(heap)
            for j in factor:
                ugly = curr * j
                if ugly not in seen:
                    seen.add(ugly)
                    heapq.heappush(heap, ugly)
        return heapq.heappop(heap)

# 前 K 个高频元素（亚马逊在半年内面试中常考）
# python的heapq有一个nlargest函数可以直接用，也可以用Counter之后再做一个sort
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        if k == len(nums):
            return nums
        res = []
        countMap = collections.Counter(nums)
        return [i[0] for i in sorted(list(countMap.items()), key=lambda x: x[1], reverse=True)[:k]]


