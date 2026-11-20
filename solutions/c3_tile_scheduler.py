"""C3 · Tile scheduling: wave quantization, raster swizzle, persistent CTAs and Stream-K — reference solution.

kernel 内部越快，“哪个 SM 算哪个 tile”就越重要。三件事：
1. wave quantization：132 个 SM、200 个 tile，第二波只有 68 个 tile，SM 利用率 76%。
2. L2 局部性：同时在跑的那一波 tile 共用越多 A 行块 / B 列块，L2 命中越高。CUTLASS 的
   raster order + swizzle 把一波 tile 排成“矮胖的方块”而不是“一整行”。
3. persistent kernel：grid = SM 数，每个 CTA 循环取 tile；Stream-K 再把 K 维也切开，
   让每个 SM 分到几乎一样多的 MAC 迭代，尾 tile 由多个 SM 各算一段 K 再做 fixup（部分和归约）。

- wave_stats(num_tiles, num_sms, ctas_per_sm=1) -> dict：slots = num_sms·ctas_per_sm；
  {"waves": ceil(num_tiles/slots), "last_wave": 最后一波的 tile 数, "efficiency": num_tiles / (waves·slots)}；
  num_tiles = 0 时 waves = 0、last_wave = 0、efficiency = 1.0。
- raster_order(m_tiles, n_tiles, swizzle=1) -> list[(m, n)]：把 m 方向按 swizzle 行一组切成条带
  （最后一条可以较短），条带按 m 递增；条带内 n 从 0 到 n_tiles−1，每个 n 再按 m 递增走完条带内的行。
  swizzle = 1 就是行优先。swizzle < 1 抛 ValueError。
- wave_unique_bytes(order, wave, a_tile_bytes, b_tile_bytes) -> int：把 order 按每 wave 个切成若干波，
  每波读 “不同 m 的个数 × a_tile_bytes + 不同 n 的个数 × b_tile_bytes”，返回各波之和
  （假设 L2 只能留住当前这一波的数据）。
- persistent_static(num_tiles, num_sms) -> list[list[int]]：SM s 依次处理 tile s, s+num_sms, …。
- stream_k(num_tiles, k_iters, num_sms) -> list[list[(tile, k_begin, k_end)]]：总迭代数
  T = num_tiles·k_iters 按 SM 平分：SM s 负责全局迭代 [s·T // num_sms, (s+1)·T // num_sms)；
  全局迭代 g 属于 tile g // k_iters 的第 g % k_iters 步。把每个 SM 的区间按 tile 边界切成段。
- fixup_tiles(assign) -> dict[int, int]：被不止一个 SM 分段处理的 tile → 参与的 SM 个数。
- makespan(assign) -> int：所有 SM 中最多的迭代数（每段 k_end − k_begin 之和）。

API
    wave_stats(num_tiles, num_sms, ctas_per_sm=1) -> dict
    raster_order(m_tiles, n_tiles, swizzle=1) -> list[tuple[int, int]]
    wave_unique_bytes(order, wave, a_tile_bytes, b_tile_bytes) -> int
    persistent_static(num_tiles, num_sms) -> list[list[int]]
    stream_k(num_tiles, k_iters, num_sms) -> list[list[tuple[int, int, int]]]
    fixup_tiles(assign) -> dict[int, int]
    makespan(assign) -> int
"""
from __future__ import annotations

import math
from collections import defaultdict


def wave_stats(num_tiles: int, num_sms: int, ctas_per_sm: int = 1) -> dict:
    """几波、最后一波几个 tile、整体利用率。"""
    slots = num_sms * ctas_per_sm
    if num_tiles == 0:
        return {"waves": 0, "last_wave": 0, "efficiency": 1.0}
    waves = math.ceil(num_tiles / slots)
    last = num_tiles - (waves - 1) * slots
    # 132 SM、200 tile：2 波，第二波 68 个，利用率 200 / 264 = 76%。
    # 这个损失和 kernel 写得多好无关，只能靠换 tile 大小、split-K / Stream-K 来补。
    return {"waves": waves, "last_wave": last, "efficiency": num_tiles / (waves * slots)}


def raster_order(m_tiles: int, n_tiles: int, swizzle: int = 1) -> list[tuple[int, int]]:
    """按 swizzle 行一组的条带，条带内按列走。"""
    if swizzle < 1:
        raise ValueError("swizzle must be >= 1")
    # 行优先：一波 132 个 tile 可能是一整行再多一点 —— 用到 1~2 个 A 行块，却要 132 个不同的 B 列块。
    # 条带化后一波是 swizzle × (132 / swizzle) 的小方块，A、B 两边都只读十几块，L2 能装下、能复用。
    out = []
    for m0 in range(0, m_tiles, swizzle):
        rows = range(m0, min(m0 + swizzle, m_tiles))
        for n in range(n_tiles):
            for mm in rows:
                out.append((mm, n))
    return out


def wave_unique_bytes(order, wave: int, a_tile_bytes: int, b_tile_bytes: int) -> int:
    """每波读的不同 A 块 + 不同 B 块的字节数，求和。"""
    total = 0
    for i in range(0, len(order), wave):
        chunk = order[i:i + wave]
        total += len({m for m, _ in chunk}) * a_tile_bytes + len({n for _, n in chunk}) * b_tile_bytes
    return total


def persistent_static(num_tiles: int, num_sms: int) -> list[list[int]]:
    """persistent kernel 的静态轮转。"""
    # grid 只有 num_sms 个 CTA：省掉每个 tile 一次的 CTA 启动 / 初始化（barrier、TMEM 分配、descriptor 预取），
    # epilogue 还能和下一个 tile 的 mainloop 重叠。但 tile 数不能被 SM 数整除时，尾巴照样不齐。
    return [list(range(s, num_tiles, num_sms)) for s in range(num_sms)]


def stream_k(num_tiles: int, k_iters: int, num_sms: int) -> list[list[tuple[int, int, int]]]:
    """按 MAC 迭代平分给各 SM，区间在 tile 边界处切开。"""
    T = num_tiles * k_iters
    out = []
    for s in range(num_sms):
        g0, g1 = s * T // num_sms, (s + 1) * T // num_sms
        segs = []
        g = g0
        while g < g1:
            tile, k = divmod(g, k_iters)
            end = min(g1, (tile + 1) * k_iters)
            segs.append((tile, k, k + (end - g)))
            g = end
        out.append(segs)
    # 每个 SM 的迭代数相差至多 1，所以没有“尾波”。代价：被切开的 tile 要把部分和写到 workspace，
    # 由最后一段（或一个专门的 SM）读回来相加——CUTLASS 把这叫 fixup / reduction。
    # CUTLASS 实际默认用 “hybrid”：前面整波用普通 data-parallel，只把最后不满的一两波交给 Stream-K，
    # 这样只有少数 tile 需要 fixup。
    return out


def fixup_tiles(assign) -> dict[int, int]:
    """被多个 SM 分段计算的 tile → SM 个数。"""
    owners = defaultdict(set)
    for s, segs in enumerate(assign):
        for tile, _, _ in segs:
            owners[tile].add(s)
    return {t: len(o) for t, o in sorted(owners.items()) if len(o) > 1}


def makespan(assign) -> int:
    """最忙的 SM 的迭代数。"""
    return max((sum(e - b for _, b, e in segs) for segs in assign), default=0)
