"""B4 · Warp specialization on Hopper: Pingpong vs Cooperative, and the setmaxnreg register split — reference solution.

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
    out = []
    tc_free = 0.0                 # 顺序 barrier：上一个 tile 的 mainloop 结束时刻
    wg_free = [0.0, 0.0]          # 每个 WG 上一次 epilogue 结束时刻
    for i in range(n_tiles):
        wg = i % 2
        # 两个约束：Tensor Core 要轮到我（上一个 tile 的 mainloop 完了），我自己也要空下来
        # （上一个 tile 的 epilogue 写完了，累加器寄存器可以重用）。
        s = max(tc_free, wg_free[wg])
        e = s + ML
        out.append({"tile": i, "wg": wg, "ml_start": s, "ml_end": e, "ep_start": e, "ep_end": e + EP})
        tc_free = e
        wg_free[wg] = e + EP
    # EP ≤ ML 时 Tensor Core 永不空闲（除了最后一个 epilogue）；EP > ML 时每个 tile 空出 EP − ML。
    # 代价：每个 WG 的 tile 只有 cooperative 的一半大（累加器寄存器有限），所以 tile 强度更低，
    # 对 SMEM 带宽要求更高——K 大、epilogue 相对短时 cooperative 往往更快。
    return out


def cooperative_timeline(n_tiles: int, ML: float, EP: float) -> list[dict]:
    """两个 WG 合作同一个 tile：mainloop 之后一起做 epilogue，Tensor Core 空着。"""
    out = []
    t = 0.0
    for i in range(n_tiles):
        s, e = t, t + ML
        out.append({"tile": i, "wg": -1, "ml_start": s, "ml_end": e, "ep_start": e, "ep_end": e + EP / 2})
        t = e + EP / 2
    return out


def makespan(tl: list[dict]) -> float:
    """最后一个 epilogue 结束的时刻。"""
    return max((x["ep_end"] for x in tl), default=0)


def tc_utilization(tl: list[dict]) -> float:
    """Tensor Core 忙碌时间 / 总时间。"""
    span = makespan(tl)
    return sum(x["ml_end"] - x["ml_start"] for x in tl) / span if span else 0.0


def max_consumer_regs(load_regs: int, n_consumers: int, regs_per_sm: int = 65536) -> int:
    """setmaxnreg 之后 consumer 每线程最多能拿多少寄存器。"""
    if load_regs % 8 or not 24 <= load_regs <= 256:
        # setmaxnreg 的操作数必须是 [24, 256] 内 8 的倍数，否则 ptxas 直接报错。
        raise ValueError("setmaxnreg count must be a multiple of 8 in [24, 256]")
    # 一个 warpgroup = 128 线程。CUTLASS pingpong：producer 40、两个 consumer 232：
    # 128·(40 + 2·232) = 64512 ≤ 65536。producer 降到 24 时 consumer 能到 240（HeavyRegisterPressure）。
    per_thread = regs_per_sm // 128 - load_regs
    x = per_thread // n_consumers
    return min(256, x // 8 * 8)
