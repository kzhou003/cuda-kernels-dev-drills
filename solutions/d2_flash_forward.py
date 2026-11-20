"""D2 · FlashAttention forward tiling: Q-block outer loop, causal block skipping and HBM traffic — reference solution.

FA1 → FA2 的算法完全一样（D1 的递推），改的是循环顺序和并行方式：
- FA1：外层循环 K/V 块、内层循环 Q 块，所以每处理一块 K/V，所有 Q 块的 O、m、l 都要从 HBM 读出来、
  更新、再写回去。
- FA2：外层是 Q 块（一个 CTA 一块），内层扫 K/V，O 一直留在寄存器里，最后写一次。
  外层可以按 Q 块并行，长序列时也能占满所有 SM。
causal 时，完全在对角线上方的块直接跳过（几乎省一半计算），只有跨对角线的块需要逐元素遮罩。
这道题在 numpy 上写出 FA2 的分块前向，并计算两种循环顺序的 HBM 流量。

- flash_forward(Q, K, V, scale, Br, Bc, causal=False) -> (O, lse, stats)：
  Q (nq, d)、K (nk, d)、V (nk, dv)。对每个 Q 块 [i0, i0+Br) 用 D1 的 m / l / acc 递推扫 K 块 [j0, j0+Bc)。
  causal 用右下对齐：query i 能看到 key j 当且仅当 j ≤ i + (nk − nq)。
  一个 (Q 块, K 块) 若块内**所有**元素都被遮住，就跳过（不计算），否则计算并对被遮住的元素置 −inf。
  stats = {"computed": 计算过的块数, "skipped": 跳过的块数}。完全被遮住的 query 行输出 0、lse = −inf。
  返回 lse 用自然对数。
- causal_blocks(nq, nk, Br, Bc) -> int：causal 时需要计算的块数（与 flash_forward 的 computed 一致）。
- hbm_bytes(N, d, Br, Bc, order, elem_bytes=2) -> int：nq = nk = N、dv = d 时，只数 Q、K、V、O 的读写
  （忽略 m、l、LSE）。Tr = ceil(N/Br)，Tc = ceil(N/Bc)。
  "fa1"：K、V 各读一次；Q 读 Tc 次；O 读 Tc 次、写 Tc 次 → (2 + Tc + 2·Tc)·N·d·elem_bytes。
  "fa2"：Q 读一次、O 写一次；K、V 各读 Tr 次 → (2 + 2·Tr)·N·d·elem_bytes。
  其他 order 抛 ValueError。

API
    flash_forward(Q, K, V, scale, Br, Bc, causal=False) -> tuple[np.ndarray, np.ndarray, dict]
    causal_blocks(nq, nk, Br, Bc) -> int
    hbm_bytes(N, d, Br, Bc, order, elem_bytes=2) -> int
"""
from __future__ import annotations

import math

import numpy as np


def _block_visible(i0, i1, j0, j1, off):
    # 块里最“靠下靠左”的元素是 (i1-1, j0)：它都看不见，整块都看不见。
    return j0 <= (i1 - 1) + off


def _block_full(i0, i1, j0, j1, off):
    # 最“靠上靠右”的元素 (i0, j1-1) 都看得见，整块都不用遮罩（FA2 的 is_even / no-mask 快路径）。
    return (j1 - 1) <= i0 + off


def flash_forward(Q, K, V, scale: float, Br: int, Bc: int, causal: bool = False):
    """FA2 风格：Q 块在外层，K/V 块在内层，O 留在“寄存器”里。"""
    Q, K, V = (np.asarray(a, dtype=np.float64) for a in (Q, K, V))
    nq, nk = Q.shape[0], K.shape[0]
    off = nk - nq
    O = np.zeros((nq, V.shape[1]))
    lse = np.full(nq, -np.inf)
    computed = skipped = 0
    for i0 in range(0, nq, Br):
        i1 = min(i0 + Br, nq)
        q = Q[i0:i1]
        m = np.full(i1 - i0, -np.inf)
        l = np.zeros(i1 - i0)
        acc = np.zeros((i1 - i0, V.shape[1]))
        for j0 in range(0, nk, Bc):
            j1 = min(j0 + Bc, nk)
            if causal and not _block_visible(i0, i1, j0, j1, off):
                # 真实 kernel 不是“跳过”，而是根本不进入循环：n_block_max 由对角线算出来
                # （FA2 flash_fwd_kernel.h 里 n_block_max = min(.., ceil((m_block+1)*kBlockM + seqlen_k - seqlen_q) / kBlockN)）。
                skipped += 1
                continue
            computed += 1
            S = scale * q @ K[j0:j1].T
            if causal and not _block_full(i0, i1, j0, j1, off):
                ii = np.arange(i0, i1)[:, None]
                jj = np.arange(j0, j1)[None, :]
                S = np.where(jj <= ii + off, S, -np.inf)
            m_new = np.maximum(m, S.max(axis=1))
            safe = np.where(np.isfinite(m_new), m_new, 0.0)
            alpha = np.exp(np.where(np.isfinite(m), m - safe, -np.inf))
            P = np.exp(S - safe[:, None])
            l = l * alpha + P.sum(axis=1)
            acc = acc * alpha[:, None] + P @ V[j0:j1]
            m = m_new
        ok = l > 0
        O[i0:i1] = np.where(ok[:, None], acc / np.where(ok, l, 1.0)[:, None], 0.0)
        lse[i0:i1] = np.where(ok, m + np.log(np.where(ok, l, 1.0)), -np.inf)
    return O, lse, {"computed": computed, "skipped": skipped}


def causal_blocks(nq: int, nk: int, Br: int, Bc: int) -> int:
    """causal 下需要计算的 (Q 块, K 块) 数。"""
    off = nk - nq
    n = 0
    for i0 in range(0, nq, Br):
        i1 = min(i0 + Br, nq)
        for j0 in range(0, nk, Bc):
            if _block_visible(i0, i1, j0, min(j0 + Bc, nk), off):
                n += 1
    return n


def hbm_bytes(N: int, d: int, Br: int, Bc: int, order: str, elem_bytes: int = 2) -> int:
    """两种循环顺序下 Q、K、V、O 的 HBM 流量。"""
    Tr, Tc = math.ceil(N / Br), math.ceil(N / Bc)
    if order == "fa1":
        # 外层 K/V：每一块 K/V 都要把全部 Q、O 过一遍，O 还要写回去。
        return (2 + Tc + 2 * Tc) * N * d * elem_bytes
    if order == "fa2":
        # 外层 Q：K/V 被每个 Q 块读一遍（实际上大多命中 L2），Q 和 O 各一次。
        return (2 + 2 * Tr) * N * d * elem_bytes
    raise ValueError("order must be 'fa1' or 'fa2'")
