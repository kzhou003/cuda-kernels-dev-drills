"""B1 · cp.async multistage pipeline: commit / wait_group and the K-loop timeline — reference solution.

Ampere 的 `cp.async` 让线程发出一次 global → shared 拷贝后立刻去干别的；拷贝按“组”提交
（`cp.async.commit_group`），再用 `cp.async.wait_group N` 等到“除最新 N 组以外”的组全部落地。
CUTLASS 的 sm80 multistage 主循环就是：prologue 先发 Stages−1 块，之后每轮
“等最老的一块 → 发最新的一块 → 算”。这道题实现这套组语义、生成主循环的操作序列，并在假时钟上
算出总时间，看 stage 数怎么把 load 延迟藏起来。

CpAsyncQueue（模拟一个线程的 cp.async 状态）：
- cp_async(tile)：把 tile 加进“当前未提交的组”。
- commit_group()：把当前组（可以是空组）封口，放到已提交组的末尾。
- wait_group(n)：除最新的 n 个已提交组外，其余已提交组全部完成；**未提交**的拷贝不受影响。
  返回此时所有已落地 tile 的集合（set）。n < 0 抛 ValueError。
- resident：已落地 tile 的集合。

multistage_ops(T, S) -> list[tuple]：T 个 K 块、S 个 stage 的操作序列，元素为
("load", t) / ("commit",) / ("wait", n) / ("mma", t)。
- S == 1（完全同步）：每个 k：load k, commit, wait 0, mma k。
- S ≥ 2：prologue 对 s = 0..S−2：若 s < T 则 load s；然后**总是** commit。
  主循环对 k = 0..T−1：wait S−2；若 k+S−1 < T 则 load k+S−1；然后**总是** commit；最后 mma k。
- S < 1 抛 ValueError。

simulate(T, S, L, C) -> dict：按 multistage_ops 在假时钟上执行。load 在当前时刻发出、L 个周期后落地，
发出本身不花时间；commit 不花时间；wait n 把时钟推进到需要完成的那些组里最晚落地的时刻（若更晚）；
mma 花 C 个周期。返回 {"total": 结束时刻, "mma_start": [每块 mma 的开始时刻], "util": T·C / total}。

API
    CpAsyncQueue()          .cp_async(tile) .commit_group() .wait_group(n) -> set .resident
    multistage_ops(T, S) -> list[tuple]
    simulate(T, S, L, C) -> dict
"""
from __future__ import annotations


class CpAsyncQueue:
    def __init__(self):
        """空队列：没有已提交组，没有已落地 tile。"""
        self._open: list[int] = []
        self._groups: list[list[int]] = []    # 已提交、尚未确认完成的组（老 → 新）
        self.resident: set[int] = set()

    def cp_async(self, tile: int) -> None:
        """把 tile 放进当前未提交的组。"""
        self._open.append(tile)

    def commit_group(self) -> None:
        """封口当前组（可以为空）。"""
        # 空组也要提交：wait_group 数的是“组”，不是“拷贝”。主循环尾部没有新块可发时如果不提交，
        # 组的数量就少了，wait_group<S-2> 等的就不再是你以为的那一块。
        self._groups.append(self._open)
        self._open = []

    def wait_group(self, n: int) -> set[int]:
        """除最新 n 组外全部完成；返回已落地 tile 集合。"""
        if n < 0:
            raise ValueError("wait_group n must be >= 0")
        # 只看已提交的组：还没 commit 的拷贝，哪怕 n = 0 也不保证完成——
        # 这是“我明明 wait_all 了为什么数据还是旧的”的常见原因（忘了 commit）。
        done = self._groups[:-n] if n else self._groups
        for g in done:
            self.resident.update(g)
        self._groups = self._groups[len(done):]
        return set(self.resident)


def multistage_ops(T: int, S: int) -> list[tuple]:
    """CUTLASS sm80 multistage 主循环的操作序列。"""
    if S < 1:
        raise ValueError("need at least one stage")
    ops: list[tuple] = []
    if S == 1:
        for k in range(T):
            ops += [("load", k), ("commit",), ("wait", 0), ("mma", k)]
        return ops
    # prologue：先把 S-1 块“在路上”。之后每轮发一块、用一块，在路上的块数保持不变。
    for s in range(S - 1):
        if s < T:
            ops.append(("load", s))
        ops.append(("commit",))
    for k in range(T):
        # 已提交组数 = (S-1) + k，wait S-2 → 前 k+1 组完成 → tile 0..k 都在 SMEM 里。
        ops.append(("wait", S - 2))
        # 新块写进 slot (k+S-1) % S == (k-1) % S，也就是上一轮刚算完的那块。
        # 真实 kernel 在这里还有一次 __syncthreads()：保证所有 warp 都读完了那个 slot 才覆盖它。
        if k + S - 1 < T:
            ops.append(("load", k + S - 1))
        ops.append(("commit",))
        ops.append(("mma", k))
    return ops


def simulate(T: int, S: int, L: int, C: int) -> dict:
    """在假时钟上执行 multistage_ops，返回总时间、每块 mma 开始时刻和利用率。"""
    now = 0
    open_done: list[int] = []          # 当前未提交组里各拷贝的落地时刻
    groups: list[list[int]] = []       # 已提交、未等待的组
    mma_start = []
    for op in multistage_ops(T, S):
        if op[0] == "load":
            open_done.append(now + L)
        elif op[0] == "commit":
            groups.append(open_done)
            open_done = []
        elif op[0] == "wait":
            n = op[1]
            done = groups[:-n] if n else groups
            for g in done:
                if g:
                    now = max(now, max(g))
            groups = groups[len(done):]
        else:
            mma_start.append(now)
            now += C
    # 稳态下每轮的周期 ≈ max(C, L / (S-1))：一块 load 提前 S-1 轮发出，
    # 只要 (S-1)·C ≥ L，延迟就被完全藏住，Tensor Core 只等第一块。
    return {"total": now, "mma_start": mma_start, "util": T * C / now if now else 0.0}
