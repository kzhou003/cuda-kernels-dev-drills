"""B4 · Warp specialization on Hopper: Pingpong vs Cooperative, and the setmaxnreg register split — your implementation (run ./check.sh b4).

H100 的 GEMM 不再是“每个 warp 都既搬又算”：一个 producer warpgroup 只发 TMA，两个 consumer
warpgroup 只做 wgmma 和 epilogue。CUTLASS 有两种 consumer 排法：
- Cooperative：两个 consumer 合作算同一个（更大的）tile，一起做 mainloop、一起做 epilogue；
  epilogue 期间 Tensor Core 空着。
- Pingpong：两个 consumer 各算各的 tile，用一个“顺序 barrier”轮流占用 Tensor Core；
  一个做 epilogue 的时候，另一个正好在跑 mainloop。
再加上 `setmaxnreg`：producer 只需要很少寄存器，把它降下来，consumer 就能升到 232 个。

时间模型（单位任意，整数）：
- 一个“单位 tile”的 mainloop 占用 Tensor Core ML，epilogue（累加器 → BF16 → SMEM → TMA store）
  花 EP，epilogue 不用 Tensor Core。Tensor Core 同一时刻只服务一个 warpgroup。
- pingpong_timeline(n_tiles, ML, EP)：tile i 给 WG i % 2。tile i 的 mainloop 开始时刻 =
  max(tile i−1 的 mainloop 结束, 同一 WG 上一个 tile 的 epilogue 结束)（tile 0 从 0 开始）；
  mainloop 结束后该 WG 立即做 epilogue。
- cooperative_timeline(n_tiles, ML, EP)：每个单位 tile 由两个 WG 合作，mainloop 占 Tensor Core ML，
  epilogue 两个 WG 各做一半、花 EP / 2；下一个 tile 的 mainloop 在上一个 epilogue 结束后开始。
- 两个函数都返回按 tile 排序的列表，每项是字典
  {"tile", "wg", "ml_start", "ml_end", "ep_start", "ep_end"}；cooperative 的 "wg" 记为 -1（两个都参与）。
- makespan(tl) = 最后一个 ep_end（空列表为 0）；tc_utilization(tl) = Σ(ml_end − ml_start) / makespan。

寄存器（每个 warpgroup 128 线程，一个 SM 65536 个 32-bit 寄存器）：
- max_consumer_regs(load_regs, n_consumers, regs_per_sm=65536)：producer 降到 load_regs 后，每个
  consumer 线程最多能升到多少：满足 128·(load_regs + n_consumers·x) ≤ regs_per_sm 的、
  不超过 256 的最大 8 的倍数 x。load_regs 必须是 [24, 256] 内 8 的倍数，否则 ValueError。

API
    pingpong_timeline(n_tiles, ML, EP) -> list[dict]
    cooperative_timeline(n_tiles, ML, EP) -> list[dict]
    makespan(tl) -> float
    tc_utilization(tl) -> float
    max_consumer_regs(load_regs, n_consumers, regs_per_sm=65536) -> int
"""
from __future__ import annotations


def pingpong_timeline(n_tiles: int, ML: float, EP: float) -> list[dict]:
    """两个 consumer WG 轮流占用 Tensor Core。"""
    raise NotImplementedError("pingpong_timeline")


def cooperative_timeline(n_tiles: int, ML: float, EP: float) -> list[dict]:
    """两个 WG 合作同一个 tile：mainloop 之后一起做 epilogue，Tensor Core 空着。"""
    raise NotImplementedError("cooperative_timeline")


def makespan(tl: list[dict]) -> float:
    """最后一个 epilogue 结束的时刻。"""
    raise NotImplementedError("makespan")


def tc_utilization(tl: list[dict]) -> float:
    """Tensor Core 忙碌时间 / 总时间。"""
    raise NotImplementedError("tc_utilization")


def max_consumer_regs(load_regs: int, n_consumers: int, regs_per_sm: int = 65536) -> int:
    """setmaxnreg 之后 consumer 每线程最多能拿多少寄存器。"""
    raise NotImplementedError("max_consumer_regs")
