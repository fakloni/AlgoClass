# 64. 最小路径和
# DP with 1-D array
# DP在这里可以用是因为只能往右和往下走，如果四个方向都可以走的话DP就会搞错了
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0
        m, n = len(grid), len(grid[0])
        dp = [0] * n
        for i in range(m-1, -1, -1):
            for j in range(n-1, -1, -1):
                if i == m-1 and j != n-1:
                    dp[j] = grid[i][j] + dp[j+1]
                elif i != m-1 and j == n-1:
                    dp[j] += grid[i][j]
                elif i != m-1 and j != n-1:
                    dp[j] = min(dp[j], dp[j+1]) + grid[i][j]
                else:
                    dp[j] = grid[i][j]
        return dp[0]

# 91. 解码方法
# DP解决，时间空间复杂度O(n)
class Solution:
    def numDecodings(self, s: str) -> int:
        dp = [0] * (len(s) + 1)
        dp[0] = 1
        dp[1] = 1 if s[0] != '0' else 0
        for i in range(2, len(dp)):
            if s[i-1] != '0':
                dp[i] = dp[i - 1]
            two_digit = int(s[i-2:i])
            if 10 <= two_digit <= 26:
                dp[i] += dp[i - 2]
        return dp[-1]


# 221. 最大正方形
# DP解决，时间空间复杂度O(mn)
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        if not matrix or not matrix[0]:
            return 0
        m, n = len(matrix), len(matrix[0])
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        maxlen = 0
        for i in range(1, m+1):
            for j in range(1, n+1):
                if matrix[i-1][j-1] == '1':
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                    maxlen = max(maxlen, dp[i][j])
        return maxlen * maxlen


# 621. 任务调度器
# 用的方法不算DP，可以直接用数学解决
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        frequencies = [0] * 26
        for t in tasks:
            frequencies[ord(t) - ord('A')] += 1
        frequencies.sort()
        f_max = frequencies.pop()
        idle_time = (f_max - 1) * n

        while frequencies and idle_time > 0:
            idle_time -= min(f_max - 1, frequencies.pop())
        idle_time = max(0, idle_time)
        return idle_time + len(tasks)

class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        frequencies = [0] * 26
        for t in tasks:
            frequencies[ord(t) - ord('A')] += 1
        f_max = max(frequencies)
        n_max = frequencies.count(f_max)
        return max(len(tasks), (f_max - 1) * (n + 1) + n_max)

# 647. 回文子串
# 时间空间复杂度 O(n^2)
class Solution:
    def countSubstrings(self, s: str) -> int:
        ans, n = 0, len(s)
        if n <= 0:
            return 0
        dp = [[False] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = True
        ans += n
        for i in range(n-1):
            if s[i] == s[i + 1]:
                dp[i][i + 1] = True
                ans += 1
        for length in range(3, n + 1):
            i = 0
            for j in range(i + length - 1, n):
                dp[i][j] = dp[i + 1][j - 1] and (s[i] == s[j])
                if dp[i][j]:
                    ans += 1
                i += 1
        return ans


# ---------------------------HARD---------------------------------------
# 32. 最长有效括号
# 时间空间复杂度O(N)
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        dp = [0] * len(s)
        maxans = 0
        for i in range(1, len(s)):
            if s[i] == ')':
                # if s[i - 1] == '(':
                #     dp[i] = dp[i - 2] + 2 if i - 2 >= 0 else 2
                if i - dp[i - 1] - 1 >= 0 and s[i - dp[i - 1] - 1] == "(":
                    dp[i] = dp[i - dp[i - 1] - 2] + dp[i - 1] + 2 if i - dp[i - 1] - 2 >= 0 else dp[i - 1] + 2
                maxans = max(maxans, dp[i])
        return maxans

# 363. 矩形区域不超过 K 的最大数值和
from sortedcontainers import SortedList


class Solution:
    def maxSumSubmatrix(self, matrix: List[List[int]], k: int) -> int:
        ans = float('-inf')
        m, n = len(matrix), len(matrix[0])

        for i in range(m):
            total = [0] * n
            for j in range(i, m):
                for c in range(n):
                    total[c] += matrix[j][c]
                totalSet = SortedList([0])
                s = 0
                for v in total:
                    s += v
                    lb = totalSet.bisect_left(s - k)
                    if lb != len(totalSet):
                        ans = max(ans, s - totalSet[lb])
                    totalSet.add(s)
        return ans


# 403. 青蛙过河
class Solution:
    def canCross(self, stones: List[int]) -> bool:
        map = {i : set() for i in stones}
        map.get(0).add(0)
        for i in range(len(stones)):
            for k in map.get(stones[i]):
                for step in (k - 1, k, k + 1):
                    if step > 0 and stones[i] + step in map:
                        map.get(stones[i] + step).add(step)
        return len(map.get(stones[-1])) > 0

# 410. 分割数组的最大值
# DP会TLE
# 记：想好DP的transition function，但是不要去人肉想矩阵怎么更新
class Solution:
    def splitArray(self, nums: List[int], m: int) -> int:
        n = len(nums)
        f = [[float('inf')] * (m + 1) for _ in range(n + 1)]
        sub = [0] * (n + 1)
        for i in range(n):
            sub[i + 1] = sub[i] + nums[i]
        f[0][0] = 0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                for k in range(j - 1, i):
                    f[i][j] = min(f[i][j], max(f[k][j - 1], sub[i] - sub[k]))
        return f[n][m]


# 552. 学生出勤记录 II
# DP方法，空间复杂O(1)，时间复杂O(N)
class Solution:
    def checkRecord(self, n: int) -> int:
        MOD = 10 ** 9 + 7
        dp = [[0, 0, 0], [0, 0, 0]]
        dp[0][0] = 1
        for i in range(1, n + 1):
            dpnew = [[0, 0, 0], [0, 0, 0]]

            # end with P
            for j in range(0, 2):
                for k in range(0, 3):
                    dpnew[j][0] = (dpnew[j][0] + dp[j][k]) % MOD
            # end with A
            for k in range(0, 3):
                dpnew[1][0] = (dpnew[1][0] + dp[0][k]) % MOD
            # end with L
            for j in range(0, 2):
                for k in range(1, 3):
                    dpnew[j][k] = (dpnew[j][k] + dp[j][k - 1]) % MOD
            dp = dpnew
        total = 0
        for j in range(0, 2):
            for k in range(0, 3):
                total += dp[j][k]
        return total % MOD


# 76. 最小覆盖子串
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if not t or not s:
            return ""

        dict_t = Counter(t)
        required = len(dict_t)
        l, r = 0, 0
        formed = 0
        window_counts = collections.defaultdict(int)
        ans = float("inf"), None, None

        while r < len(s):
            char = s[r]
            window_counts[char] += 1
            if char in dict_t and window_counts[char] == dict_t[char]:
                formed += 1

            while l <= r and formed == required:
                char = s[l]
                if r - l + 1 < ans[0]:
                    ans = (r - l + 1, l, r)
                window_counts[char] -= 1
                if char in dict_t and window_counts[char] < dict_t[char]:
                    formed -= 1
                l += 1
            r += 1
        return "" if ans[0] == float('inf') else s[ans[1]: ans[2] + 1]


# 312. 戳气球
# DP top-down approach
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        if len(nums) > 1 and len(set(nums)) == 1:
            return (nums[0] ** 3) * (len(nums) - 2) + nums[0] ** 2 + nums[0]

        nums = [1] + nums + [1]

        @lru_cache(None)
        def dp(left, right):
            if right - left < 0:
                return 0
            result = 0
            for i in range(left, right + 1):
                gain = nums[left - 1] * nums[i] * nums[right + 1]
                remaining = dp(left, i - 1) + dp(i + 1, right)
                result = max(result, remaining + gain)
            return result

        return dp(1, len(nums) - 2)