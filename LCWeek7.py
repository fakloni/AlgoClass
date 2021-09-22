# 70. 爬楼梯
# 优化后的动态规划搜索，时间复杂度是O(n)，空间复杂度是O(1)
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 2:
            return n
        first = 1
        second = 2
        for i in range(3, n + 1):
            third = first + second
            first = second
            second = third
        return second

# 22. 括号生成
# 回溯，逻辑清晰
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        def backtrack(left, right, cur):
            if left == right == n:
                output.append("".join(cur))
                return
            if left < n:
                cur.append("(")
                backtrack(left + 1, right, cur)
                cur.pop()
            if right < left:
                cur.append(")")
                backtrack(left, right + 1, cur)
                cur.pop()
        output = []
        backtrack(0, 0, [])
        return output

# 36. 有效的数独
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        cols = [set(map(str, range(1, 10))) for _ in range(9)]
        rows = [set(map(str, range(1, 10))) for _ in range(9)]
        blocks = [set(map(str, range(1, 10))) for _ in range(9)]

        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == '.':
                    continue
                blockNum = (i // 3) * 3 + (j // 3)
                if board[i][j] not in rows[i] & cols[j] & blocks[blockNum]:
                    return False
                rows[i].remove(board[i][j])
                cols[j].remove(board[i][j])
                blocks[blockNum].remove(board[i][j])
        return True

# 127. 单词接龙
# 双向BFS
class Solution:
    def __init__(self):
        self.alldict = collections.defaultdict(list)
        self.L = 0

    def visit(self, q, visited, other_visited):
        word, level = q.popleft()
        for i in range(self.L):
            for new_word in self.alldict[word[:i] + "*" + word[i + 1:]]:
                if new_word in other_visited:
                    return level + other_visited[new_word]
                if new_word not in visited:
                    visited[new_word] = level + 1
                    q.append((new_word, level + 1))
        return None

    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if not beginWord or not endWord or endWord not in wordList:
            return 0
        self.L = len(beginWord)
        for word in wordList:
            for i in range(self.L):
                self.alldict[word[:i] + "*" + word[i + 1:]].append(word)
        front_visited = {beginWord: 1}
        end_visited = {endWord: 1}
        front_q = collections.deque([(beginWord, 1)])
        end_q = collections.deque([(endWord, 1)])
        while front_q and end_q:
            ans = self.visit(front_q, front_visited, end_visited)
            if ans:
                return ans
            ans = self.visit(end_q, end_visited, front_visited)
            if ans:
                return ans
        return 0


# 433. 最小基因变化
# 跟wordladder题目一样
class Solution:
    def minMutation(self, start: str, end: str, bank: List[str]) -> int:
        if not start or not end or end not in bank:
            return -1
        L = len(start)
        bank = set(bank)
        front = {start}
        back = {end}
        dist = 0
        while front:
            dist += 1
            new_front = set()
            for seq in front:
                for i in range(L):
                    for gene in ["A", "C", "G", "T"]:
                        if gene != seq[i]:
                            new_seq = seq[:i] + gene + seq[i+1:]
                            if new_seq in back:
                                print(new_seq)
                                return dist
                            if new_seq in bank:
                                bank.remove(new_seq)
                                new_front.add(new_seq)
            front = new_front
            if len(back) < len(front):
                front, back = back, front
        return -1


# 51. N-Queens
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        def dfs(row, xy_sum, xy_diff, queens, valid):
            if row == n:
                valid.append(queens)
            for col in range(n):
                if col not in queens and row - col not in xy_diff and row + col not in xy_sum:
                    dfs(row + 1, xy_sum + [row + col], xy_diff + [row - col], queens + [col], valid)
        result = []
        dfs(0, [], [], [], result)
        return [['.'*i + 'Q' + '.'*(n - i - 1) for i in sol] for sol in result]


# 37. 解数独
# 回溯 Max operations needed: (9!)^9, Space complexity: O(m*n)
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        row = [set(range(1, 10)) for _ in range(9)]
        col = [set(range(1, 10)) for _ in range(9)]
        block = [set(range(1, 10)) for _ in range(9)]

        empty_pos = []
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] != '.':
                    val = int(board[i][j])
                    row[i].remove(val)
                    col[j].remove(val)
                    block[(i // 3) * 3 + j // 3].remove(val)
                else:
                    empty_pos.append((i, j))

        def backtrack(iter=0):
            if iter == len(empty_pos):
                return True
            i, j = empty_pos[iter]
            b = (i // 3) * 3 + j // 3
            for val in row[i] & col[j] & block[b]:
                board[i][j] = str(val)
                row[i].remove(val)
                col[j].remove(val)
                block[b].remove(val)
                if backtrack(iter + 1):
                    return True
                row[i].add(val)
                col[j].add(val)
                block[b].add(val)
            return False

        backtrack()