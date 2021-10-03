# ---------------------------------- EASY ----------------------------------------------

# 191. 位1的个数
# 位运算 O(1)时间空间复杂度
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n > 0:
            count += 1
            n = n & (n - 1)
        return count


# 231. 2 的幂
# 位运算 O(1)时间空间复杂度
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        if n == 0:
            return False
        return n & (n - 1) == 0


# 190. 颠倒二进制位
# 位运算 O(1)时间空间复杂度 但是位运算分治看不懂..
class Solution:
    def reverseBits(self, n):
        res = 0
        for i in range(32):
            res = (res << 1) | (n & 1)
            n >>= 1
        return res

# ----------------------------------MEDIUM----------------------------------------------
# 208. 实现 Trie (前缀树)
class Trie:

    def __init__(self):
        self.children = [None] * 26
        self.isEnd = False

    def insert(self, word: str) -> None:
        node = self
        for ch in word:
            ch = ord(ch) - ord('a')
            if not node.children[ch]:
                node.children[ch] = Trie()
            node = node.children[ch]
        node.isEnd = True

    def searchPrefix(self, prefix: str) -> "Trie":
        node = self
        for ch in prefix:
            ch = ord(ch) - ord('a')
            if not node.children[ch]:
                return None
            node = node.children[ch]
        return node

    def search(self, word: str) -> bool:
        node = self.searchPrefix(word)
        return node != None and node.isEnd

    def startsWith(self, prefix: str) -> bool:
        return self.searchPrefix(prefix) != None


# 547. 省份数量
# 用并查集，时间复杂度是O(n^3)，最坏情况需要O(n)来find parent，然后需要遍历整个matrix
class UnionFind:
    def __init__(self, grid):
        self.parent = [i for i in range(len(grid))]
        self.rank = [0] * len(grid)
        self.count = len(grid)

    def find(self, x):
        if self.parent[x] != x:
            return self.find(self.parent[x])
        return x

    def union(self, x, y):
        rootx, rooty = self.find(x), self.find(y)
        if rootx != rooty:
            if self.rank[rootx] < self.rank[rooty]:
                rootx, rooty = rooty, rootx
            self.parent[rooty] = rootx
            if self.rank[rooty] == self.rank[rootx]:
                self.rank[rootx] += 1
            self.count -= 1

    def countUnion(self):
        return self.count

class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        L = len(isConnected)
        uf = UnionFind(isConnected)
        for i in range(L):
            for j in range(L):
                if isConnected[i][j] == 1:
                    uf.union(i, j)
        return uf.countUnion()


# 200. 岛屿数量
# 并查集
class UnionFind:
    def __init__(self, matrix):
        m, n = len(matrix), len(matrix[0])
        self.parent = [None] * (m * n)
        self.rank = [0] * (m * n)
        self.count = 0
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    self.parent[n * i + j] = n * i + j
                    self.count += 1

    def find(self, x):
        if self.parent[x] != x:
            return self.find(self.parent[x])
        return x

    def union(self, x, y):
        rootx, rooty = self.find(x), self.find(y)
        if rootx != rooty:
            if self.rank[rootx] < self.rank[rooty]:
                rootx, rooty = rooty, rootx
            self.parent[rooty] = rootx
            if self.rank[rootx] == self.rank[rooty]:
                self.rank[rootx] += 1
            self.count -= 1

    def countUnion(self):
        return self.count

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid or not grid[0]:
            return 0
        uf = UnionFind(grid)
        m, n = len(grid), len(grid[0])
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1":
                    grid[i][j] = 0
                    for d in directions:
                        new_i, new_j = i + d[0], j + d[1]
                        if 0 <= new_i < m and 0 <= new_j < n and grid[new_i][new_j] == "1":
                            uf.union(i * n + j, new_i * n + new_j)
        return uf.countUnion()


# 130. 被围绕的区域
# 并查集训练
class UnionFind:
    def __init__(self, size):
        self.parent = [i for i in range(size)]

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rootx, rooty = self.find(x), self.find(y)
        if rootx != rooty:
            self.parent[rooty] = rootx

    def isConnected(self, x, y):
        return self.find(x) == self.find(y)

class Solution:
    def solve(self, board: List[List[str]]) -> None:
        if not board:
            return
        m, n = len(board), len(board[0])
        uf = UnionFind(m * n + 1)
        dummy = m * n
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O':
                    if i == 0 or i == m - 1 or j == 0 or j == n - 1:
                        uf.union(i * n + j, dummy)
                    else:
                        for nx, ny in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
                            if 0 <= nx < m and 0 <= ny < n and board[nx][ny] == 'O':
                                uf.union(i * n + j, nx * n + ny)

        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O':
                    if not uf.isConnected(i * n + j, dummy):
                        board[i][j] = "X"


# ---------------------------------- HARD ----------------------------------------------
# 212. 单词搜索 II
# 用字典树trie
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        WORD_KEY = "#"
        trie = {}
        for word in words:
            node = trie
            for letter in word:
                node = node.setdefault(letter, {})
            node[WORD_KEY] = word

        m, n = len(board), len(board[0])
        matchedWords = []

        def backtrack(x, y, parent):
            letter = board[x][y]
            curNode = parent[letter]
            matched = curNode.pop(WORD_KEY, False)
            if matched:
                matchedWords.append(matched)
            board[x][y] = "@"
            for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_x, new_y = x + d[0], y + d[1]
                if 0 <= new_x < m and 0 <= new_y < n and board[new_x][new_y] in curNode:
                    backtrack(new_x, new_y, curNode)
            board[x][y] = letter
            if not curNode:
                parent.pop(letter)

        for i in range(m):
            for j in range(n):
                if board[i][j] in trie:
                    backtrack(i, j, trie)
        return matchedWords


# 51. N 皇后
# 位运算
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:

        def generateBoard():
            board = []
            for i in range(n):
                idx = queens[i]
                board.append("." * idx + "Q" + "." * (n - idx - 1))
            return board

        def backtrack(row, cols, pie, na):
            if row == n:
                solutions.append(generateBoard())
                return
            bits = ((1 << n) - 1) & (~(cols | pie | na))
            while bits:
                position = bits & (-bits)
                bits = bits & (bits - 1)
                colNum = bin(position - 1).count("1")
                queens[row] = colNum
                backtrack(row + 1, cols | position, (pie | position) << 1, (na | position) >> 1)

        solutions = []
        queens = [None] * n
        backtrack(0, 0, 0, 0)
        return solutions


# 52. N皇后 II
# 课上题解，用位运算来解决
class Solution:
    def totalNQueens(self, n: int) -> int:
        if n < 1:
            return 1
        self.count = 0
        self.DFS(n, 0, 0, 0, 0)
        return self.count

    def DFS(self, n, row, cols, pie, na):
        if row >= n:
            self.count += 1
            return
        bits = (~(cols | pie | na)) & ((1 << n) - 1)
        while bits:
            p = bits & -bits
            bits = bits & (bits - 1)
            self.DFS(n, row + 1, cols | p, (pie | p) << 1, (na | p) >> 1)