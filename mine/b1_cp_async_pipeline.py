"""B1 · cp.async multistage pipeline: commit / wait_group and the K-loop timeline — your implementation (run ./check.sh b1).

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
        raise NotImplementedError("CpAsyncQueue.__init__")

    def cp_async(self, tile: int) -> None:
        """把 tile 放进当前未提交的组。"""
        raise NotImplementedError("CpAsyncQueue.cp_async")

    def commit_group(self) -> None:
        """封口当前组（可以为空）。"""
        raise NotImplementedError("CpAsyncQueue.commit_group")

    def wait_group(self, n: int) -> set[int]:
        """除最新 n 组外全部完成；返回已落地 tile 集合。"""
        raise NotImplementedError("CpAsyncQueue.wait_group")


def multistage_ops(T: int, S: int) -> list[tuple]:
    """CUTLASS sm80 multistage 主循环的操作序列。"""
    raise NotImplementedError("multistage_ops")


def simulate(T: int, S: int, L: int, C: int) -> dict:
    """在假时钟上执行 multistage_ops，返回总时间、每块 mma 开始时刻和利用率。"""
    raise NotImplementedError("simulate")
