# ------------------EASY------------------------------------------------------------------------------------------------
# 860 柠檬水找零
# 两个变量存储5和10元面值的数量即可
class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        five, ten = 0, 0
        for bill in bills:
            if bill == 5:
                five += 1
            elif bill == 10:
                if not five:
                    return False
                five -= 1
                ten += 1
            else:
                if five and ten:
                    five -= 1
                    ten -= 1
                elif five >= 3:
                    five -= 3
                else:
                    return False
        return True

# 买卖股票最佳时机II
# 用动态规划
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        dp = [[0] * 2 for _ in range(len(prices))]
        dp[0][0] = 0
        dp[0][1] = -prices[0]
        for i in range(1, len(prices)):
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
        return dp[-1][0]
# 用贪心算法
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                profit += prices[i] - prices[i-1]
        return profit

# 455 分发饼干
# 用贪心算法可以得到最优解，但是要先把两个input数组排序
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g.sort()
        s.sort()
        res, i, j = 0, 0, 0
        while i < len(g) and j < len(s):
            while j < len(s) and g[i] > s[j]:
                j += 1
            if j < len(s):
                res += 1
            i += 1
            j += 1
        return res

# 874 模拟行走机器人
# 根据题意编程，逻辑难度不大，在Python中用set比用list时间要快非常非常多，用list的时候出现TLE error
class Solution:
    def robotSim(self, commands: List[int], obstacles: List[List[int]]) -> int:
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]
        x, y, di, ans = 0, 0, 0, 0
        obstacleSet = set(map(tuple, obstacles))
        for command in commands:
            if command == -1:
                di = (di + 1) % 4
            elif command == -2:
                di = (di - 1) % 4
            else:
                for i in range(command):
                    if (x + dx[di], y + dy[di]) not in obstacleSet:
                        x += dx[di]
                        y += dy[di]
                ans = max(ans, x*x + y*y)
        return ans

# ------------------MEDIUM----------------------------------------------------------------------------------------------
# 127 单词接龙
# 难点，先构建一个图，图构建好之后用BFS来搜索就比较常规

class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if endWord not in wordList or not endWord or not beginWord or not wordList:
            return 0
        L = len(beginWord)
        all_combo_dict = collections.defaultdict(list)
        # Construct a word node graph
        for word in wordList:
            for i in range(L):
                all_combo_dict[word[:i] + "*" + word[i+1:]].append(word)
        # BFS
        queue = collections.deque([(beginWord, 1)])
        visited = {beginWord: True}
        while queue:
            current_word, level = queue.popleft()
            for i in range(L):
                intermediate_word = current_word[:i] + "*" + current_word[i+1:]
                for word in all_combo_dict[intermediate_word]:
                    if word == endWord:
                        return level + 1
                    if word not in visited:
                        visited[word] = True
                        queue.append((word, level + 1))
                all_combo_dict[intermediate_word] = []
        return 0


# 200 岛屿数量
# 用BFS和DFS很好理解，合并集这个需要多练习，现在理解了合并集但是代码还是要更加熟悉
# 时间空间复杂度都是O(M * N) M和N是grid的size
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if grid == None or len(grid) == 0:
            return 0
        self.DIRECTIONS = {(1, 0), (0, 1), (-1, 0), (0, -1)}
        self.row, self.col = len(grid), len(grid[0])
        res = 0
        for i in range(self.row):
            for j in range(self.col):
                if grid[i][j] == "1":
                    res += 1
                    # 两个都可以用
                    # self.bfs(grid, i, j)
                    self.dfs(grid, i, j)
        return res

    def bfs(self, grid, r, c):
        q = collections.deque([(r, c)])
        grid[r][c] = "0"
        while q:
            cur_r, cur_c = q.popleft()
            for direction in self.DIRECTIONS:
                new_r, new_c = cur_r + direction[0], cur_c + direction[1]
                if 0 <= new_r < self.row and 0 <= new_c < self.col and grid[new_r][new_c] == "1":
                    q.append((new_r, new_c))
                    grid[new_r][new_c] = "0"

    def dfs(self, grid, r, c):
        grid[r][c] = "0"
        for direction in self.DIRECTIONS:
            new_r, new_c = r + direction[0], c + direction[1]
            if 0 <= new_r < self.row and 0 <= new_c < self.col and grid[new_r][new_c] == "1":
                self.dfs(grid, new_r, new_c)


# 529 扫雷游戏
# DFS的方法，时间空间复杂度为O(nm)，在最坏情况下会遍历整个面板
class Solution:
    def updateBoard(self, board: List[List[str]], click: List[int]) -> List[List[str]]:
        x, y = click[1], click[0]
        self.DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1))
        if board[y][x] == 'M':
            board[y][x] = 'X'
        else:
            self.dfs(board, y, x)
        return board

    def dfs(self, board, y, x):
        mine_count = 0
        for direction in self.DIRECTIONS:
            newX, newY = x + direction[0], y + direction[1]
            if 0 <= newX < len(board[0]) and 0 <= newY < len(board):
                if board[newY][newX] == 'M':
                    mine_count += 1
        if mine_count != 0:
            board[y][x] = f"{mine_count}"
        else:
            board[y][x] = 'B'
            for direction in self.DIRECTIONS:
                newX, newY = x + direction[0], y + direction[1]
                if 0 <= newX < len(board[0]) and 0 <= newY < len(board):
                    if board[newY][newX] == 'E':
                        self.dfs(board, newY, newX)


# 55 跳跃游戏
# 用贪心算法非常简单可以解决，但是看到这种题目第一反应还是运用动态规划来解决
# 贪心算法的时间复杂度是O(n)，只需遍历一次数组，空间复杂度为O(1)
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        last_pos = len(nums) - 1
        for i in reversed(range(len(nums)-1)):
            if i + nums[i] >= last_pos:
                last_pos = i
        return last_pos == 0


# 33 搜索旋转排序数组
# 用二分法的话时间复杂度是O(logn)，空间复杂度是O(1)
# 难点在于设置二分界限
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] >= nums[left]:
                if target >= nums[left] and target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if target < nums[left] and target > nums[mid]:
                    left = mid + 1
                else:
                    right = mid - 1
        return -1


# 74 搜索二维矩阵
# 二分搜索，把矩阵的数字按照index来排列，然后也可以根据index算出行数和列数
# Time complexity: O(logmn), Space complexity: O(1)
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m = len(matrix)
        if m == 0:
            return False
        n = len(matrix[0])

        left, right = 0, m * n - 1
        while left <= right:
            mid = (left + right) // 2
            mid_value = matrix[mid // n][mid % n]
            if target == mid_value:
                return True
            elif target > mid_value:
                left = mid + 1
            else:
                right = mid - 1
        return False


# 153 寻找旋转排序数组中的最小值
# 二分搜索
class Solution:
    def findMin(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) // 2
            if nums[mid] >= nums[left] and nums[left] > nums[right]:
                left = mid + 1
            else:
                right = mid
        return nums[left]


# ------------------HARD------------------------------------------------------------------------------------------------

# 126 单词接龙 II
# 思路还是先构图，用一遍BFS，同时记录path
class Solution:
    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        if endWord not in wordList:
            return []
        L = len(beginWord)
        alldict = collections.defaultdict(list)
        for word in wordList:
            for i in range(L):
                alldict[word[:i] + "*" + word[i+1:]].append(word)
        queue = collections.deque([[beginWord]])
        visited = set()
        res = []
        found = False
        while queue:
            qsize = len(queue)
            for words in queue:
                    visited.add(words[-1])
            for _ in range(qsize):
                current_path = queue.popleft()
                current_word = current_path[-1]
                for i in range(L):
                    intermediate_word = current_word[:i] + "*"+ current_word[i+1:]
                    for word in alldict[intermediate_word]:
                        if word == endWord:
                            found = True
                            res.append(current_path + [word])
                        if word not in visited:
                            queue.append(current_path + [word])
            if found:
                break
        return res


# 45 跳跃游戏II
# 贪心算法，这次从前往后，但是要主要贪心的规则以及数值更新
class Solution:
    def jump(self, nums: List[int]) -> int:
            jump, currentJumpEnd, farthest = 0, 0, 0
            for i in range(len(nums) - 1):
                farthest = max(farthest, i + nums[i])
                if i == currentJumpEnd:
                    jump += 1
                    currentJumpEnd = farthest
            return jump