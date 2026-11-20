"""D3 · FA4 softmax tricks on Blackwell: exp2 on the FMA pipe and conditional rescaling — your implementation (run ./check.sh d3).

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
    raise NotImplementedError("ex2_emulation")


def softmax_attention_cond_rescale(Q, K, V, scale: float, Bc: int, threshold: float):
    """log2 域 online softmax；最大值只涨一点时跳过 rescale。"""
    raise NotImplementedError("softmax_attention_cond_rescale")


def exp_to_mma_ratio(tc_flops_per_clk: float, mufu_per_clk: float, d: int) -> float:
    """每对 (q, k)：exp 时间 / MMA 时间。"""
    raise NotImplementedError("exp_to_mma_ratio")


def balanced_emulation_fraction(mufu_per_clk: float, fma_per_clk: float, ops_per_emulated: float) -> float:
    """MUFU 和 FMA 两条管线时间相等时，挪到 FMA 上的 exp 比例。"""
    raise NotImplementedError("balanced_emulation_fraction")
