# 236 二叉树最近的公共祖先
# 用的方法是stack然后保存parent node
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':

        stack = [root]
        parent = {root: None}

        while q not in parent or p not in parent:
            node = stack.pop()
            if node.left:
                parent[node.left] = node
                stack.append(node.left)
            if node.right:
                parent[node.right] = node
                stack.append(node.right)

        ancestors = set()
        while p:
            ancestors.add(p)
            p = parent[p]
        while q not in ancestors:
            q = parent[q]
        return q

#递归方法，比较简洁，以上两种的时间空间复杂度都为O(n)，n = number of nodes
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':

        if not root or root == q or root == p:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if not left:
            return right
        if not right:
            return left
        return root


# 105 从前序与中序遍历构造二叉树
# preorder首先可以得出根，然后通过找到在inorder根的index position，可以再分成左子树和右子树
# note：递归解决子问题写出来的代码很简洁，用人肉递归来试图理解的话真的有点复杂
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        self.preorder_index = 0
        inorder_map = {v: i for i, v in enumerate(inorder)}
        def divide(left, right):
            if left > right:
                return None
            root_val = preorder[self.preorder_index]
            self.preorder_index += 1
            root = TreeNode(root_val)
            root.left = divide(left, inorder_map[root_val] - 1)
            root.right = divide(inorder_map[root_val] + 1, right)
            return root
        return divide(0, len(preorder) - 1)


# 77 组合
# 这道题一开始想的时候很复杂，不知道怎么解决，后面发现用回溯就会非常简单
# 空间复杂度是n choose k，时间复杂度是k* （n choose k）因为每个组合在找到之前都会递归k次。
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        def backtrack(first=1, cur=[]):
            if len(cur) == k:
                ans.append(cur[:])
                return
            for i in range(first, n + 1):
                cur.append(i)
                backtrack(i + 1, cur)
                cur.pop()
        ans = []
        backtrack()
        return ans


# 46 全排列
# 回溯解决，难点就在于决定回溯条件是什么，terminator很容易决定，process还有drill down这个部分比较难判断
# 空间复杂度就是O(n) 因为就只是在数组上面操作，时间复杂度是O(n*n!)
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def backtrack(first):
            if first == n:
                res.append(nums[:])
            for i in range(first, n):
                nums[first], nums[i] = nums[i], nums[first]
                backtrack(first + 1)
                nums[first], nums[i] = nums[i], nums[first]
        res = []
        n = len(nums)
        backtrack(0)
        return res


# 47 全排列II
# 同样用回溯，如果用上面那道题再加一个判断是否在res里的话，那么回溯的过程中会生成很多已经重复的排列，所以用Counter更优
# 时间空间复杂度跟上题一样
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        def backtrack(comb, counter):
            if len(comb) == len(nums):
                res.append(comb[:])
                return
            for num in counter:
                if counter[num] > 0:
                    comb.append(num)
                    counter[num] -= 1
                    backtrack(comb, counter)
                    comb.pop()
                    counter[num] += 1
        res = []
        backtrack([], Counter(nums))
        return res