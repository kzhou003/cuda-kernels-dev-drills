"""D3 · FA4 softmax tricks on Blackwell: exp2 on the FMA pipe and conditional rescaling — reference solution.

每个 (query, key) 对要做 4·d 次 MMA FLOP 和 1 次 exp。Tensor Core 每代翻倍，算 exp 的 MUFU
（每 SM 每时钟 16 次）却没变：A100 上 exp 的时间约是 MMA 的 1/4，H100 约 1/2，B200 上已经一样长。
FA4 的两个招：
1. 一部分 exp2 不用 MUFU，而是在 FMA 单元上用多项式算：2^x = 2^⌊x⌋ · 2^f，f ∈ [0, 1)，
   2^f 用 3 次多项式（Horner，3 次 FMA），2^⌊x⌋ 直接加到浮点数的指数位上。BF16 的 P 只有 8 位尾数，
   3 次多项式的误差（~1e-4）已经足够。
2. 条件 rescale：行最大值只涨了一点（log2 域里 ≤ threshold，比如 8）时，不更新 m、不 rescale O，
   让 P 的值最大到 2^threshold（FP32 装得下）。最终 O = acc / l 用的是同一个 m，结果仍然精确。
   省掉的是每块对整个 O 累加器（在 TMEM 里）的一次读改写。

给定（given）：POLY_EX2（FA4 源码里 degree 3 的系数，[c0, c1, c2, c3]）、LOG2E。

- ex2_emulation(x)：float32 输入（标量或数组），返回 float32 的 2^x 近似。步骤：x = max(x, −127)；
  j = floor(x)；f = x − j；p = ((c3·f + c2)·f + c1)·f + c0（全部 float32）；
  结果 = 把 p 的 IEEE 位模式（int32）加上 j << 23 后按 float32 解释。假设 x ≤ 127。
- softmax_attention_cond_rescale(Q, K, V, scale, Bc, threshold) -> (O, n_rescales)：
  log2 域的 online softmax：s = scale·LOG2E·QKᵀ。对每个 key 块，每一行：
    m_cand = max(m, rowmax(s_blk))；第一块：m = m_cand，不算 rescale；
    之后：delta = m − m_cand（≤ 0）。threshold > 0 且 delta ≥ −threshold 时保持 m 不变、不 rescale；
    否则 acc、l 乘 2^delta，m = m_cand，n_rescales += 1（逐行计数）。
    然后 p = 2^(s_blk − m)，l += Σp，acc += p·V_blk。
  最后 O = acc / l。用 float64 计算（np.exp2）。threshold = 0 表示每次最大值变大都 rescale
  （delta < 0 时；delta == 0 时 2^0 = 1，不计数）。
- exp_to_mma_ratio(tc_flops_per_clk, mufu_per_clk, d)：每对 (q, k) 的 exp 时间 / MMA 时间
  = (1 / mufu_per_clk) / (4·d / tc_flops_per_clk)。
- balanced_emulation_fraction(mufu_per_clk, fma_per_clk, ops_per_emulated)：把比例 f 的 exp 挪到 FMA 上，
  使 MUFU 时间 (1−f)/mufu 与 FMA 时间 f·ops/fma 相等时的 f = fma / (fma + ops·mufu)。

API
    POLY_EX2, LOG2E                                    given
    ex2_emulation(x) -> np.ndarray | np.float32
    softmax_attention_cond_rescale(Q, K, V, scale, Bc, threshold) -> tuple[np.ndarray, int]
    exp_to_mma_ratio(tc_flops_per_clk, mufu_per_clk, d) -> float
    balanced_emulation_fraction(mufu_per_clk, fma_per_clk, ops_per_emulated) -> float
"""
from __future__ import annotations

import numpy as np

POLY_EX2 = (1.0, 0.695146143436431884765625, 0.227564394474029541015625, 0.077119089663028717041015625)
LOG2E = 1.4426950408889634


def ex2_emulation(x):
    """FA4 的 ex2_emulation：整数部分进指数位，小数部分用 3 次多项式。"""
    x = np.maximum(np.asarray(x, dtype=np.float32), np.float32(-127.0))
    # FA4 里 floor 是用 “加 1.5·2^23、向下舍入” 的技巧在 FMA 管线上做的（add.rm），
    # 这样整数部分直接落在尾数低 8 位里，后面用一次整数加法拼进指数位。这里用 np.floor 表达同一件事。
    j = np.floor(x)
    f = (x - j).astype(np.float32)
    c0, c1, c2, c3 = (np.float32(c) for c in POLY_EX2)
    p = ((c3 * f + c2) * f + c1) * f + c0          # 3 次 FMA，p ∈ [1, 2)
    # p 的指数是 0（偏置 127）；加上 j << 23 就是把指数改成 j，即乘 2^j。
    # x ≥ −127 保证结果不会下溢成负的指数位（FA4 同样先 fmax(x, -127)）。
    bits = p.view(np.int32) + (j.astype(np.int32) << 23)
    out = bits.view(np.float32)
    return out if out.ndim else np.float32(out)


def softmax_attention_cond_rescale(Q, K, V, scale: float, Bc: int, threshold: float):
    """log2 域 online softmax；最大值只涨一点时跳过 rescale。"""
    Q, K, V = (np.asarray(a, dtype=np.float64) for a in (Q, K, V))
    nq, nk = Q.shape[0], K.shape[0]
    m = np.full(nq, -np.inf)
    l = np.zeros(nq)
    acc = np.zeros((nq, V.shape[1]))
    n_rescales = 0
    for j0 in range(0, nk, Bc):
        s = scale * LOG2E * (Q @ K[j0:j0 + Bc].T)
        cand = np.maximum(m, s.max(axis=1))
        if j0 == 0:
            m = cand
        else:
            delta = m - cand
            keep = (delta >= -threshold) if threshold > 0 else (delta == 0)
            # keep 的行：继续用旧的 m。新块里的 s 可能比 m 大，最多大 threshold，
            # 所以 p = 2^(s − m) ≤ 2^threshold（threshold = 8 → 256），FP32 的 acc、l 毫无压力。
            # 只要分子 acc 和分母 l 用的是同一个 m，最后相除时 2^m 抵消，结果精确。
            factor = np.where(keep, 1.0, np.exp2(delta))
            n_rescales += int((~keep).sum())
            acc *= factor[:, None]
            l *= factor
            m = np.where(keep, m, cand)
        p = np.exp2(s - m[:, None])
        l += p.sum(axis=1)
        acc += p @ V[j0:j0 + Bc]
    # FA4 里 rescale 由单独的 “correction” warpgroup 做（O 在 TMEM 里，要 tcgen05.ld / st 一遍），
    # 所以少一次 rescale 就是少一次对 TMEM 里整个 O tile 的读改写。
    return acc / l[:, None], n_rescales


def exp_to_mma_ratio(tc_flops_per_clk: float, mufu_per_clk: float, d: int) -> float:
    """每对 (q, k)：exp 时间 / MMA 时间。"""
    # 每对 (q,k)：S 的一个元素要 2d FLOP（QKᵀ），P·V 再 2d FLOP，共 4d；exp 一次。
    return (1.0 / mufu_per_clk) / (4.0 * d / tc_flops_per_clk)


def balanced_emulation_fraction(mufu_per_clk: float, fma_per_clk: float, ops_per_emulated: float) -> float:
    """MUFU 和 FMA 两条管线时间相等时，挪到 FMA 上的 exp 比例。"""
    # 真实 kernel 里 FMA 管线还要做 scale、max、类型转换等，所以 FA4 实际挪过去的比例更小，
    # 而且按 fragment 交错（config.py 里的 ex2_emu_freq / ex2_emu_start_frg）。
    return fma_per_clk / (fma_per_clk + ops_per_emulated * mufu_per_clk)
