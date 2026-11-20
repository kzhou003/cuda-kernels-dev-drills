"""C2 · Block-scaled low precision: MXFP8 / MXFP4 / NVFP4 and DeepGEMM's FP8 promotion — reference solution.

FP8 / FP4 让 Tensor Core 吞吐再翻倍、翻四倍，但 4 bit 只能表示 ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}。
能用的前提是“每一小块共享一个 scale”：MX 格式每 32 个元素一个 2 的幂（UE8M0）scale，
NVFP4 每 16 个元素一个 E4M3 scale 再乘一个全张量 FP32 scale。Blackwell 的
`tcgen05.mma.kind::mxf8f6f4 / mxf4nvf4 .block_scale` 在 Tensor Core 里直接乘这些 scale；
Hopper 没有这个能力，DeepGEMM 每 128 个 K 把 Tensor Core 的部分和取出来，在 CUDA core 上乘 scale
再加进 FP32 累加器（同时顺带解决了 FP8 wgmma 累加精度只有约 14 位的问题）。

给定（given）：E2M1_GRID、round_e4m3(x)、round_mantissa(x, bits)。

- quantize_e2m1(x)：逐元素就近取到 ±E2M1_GRID 上；正好在两个格点中间时取下标为偶数的那个
  （即尾数位为 0 的那个，round-half-to-even）；|x| > 6 饱和到 ±6；保留符号。
- mx_quantize(x, block=32, elem="e4m3")：x 是 1D、长度为 block 的倍数（否则 ValueError）。每块：
  amax = max|x|；amax == 0 时 scale = 1，否则 scale = 2^(floor(log2(amax)) − emax)，
  emax = 8（e4m3）或 2（e2m1），指数夹到 [−127, 127]。q = round_elem(x / scale)
  （e4m3 用 round_e4m3，e2m1 用 quantize_e2m1，都饱和）。返回 (q, scales)，scales 长度 = 块数。
  elem 不是这两个时 ValueError。
- nvfp4_quantize(x, block=16)：g = amax(|x|) / (448·6)（全为 0 时 g = 1）；每块
  s = round_e4m3(amax_block / (6·g))；q = quantize_e2m1(x / (s·g))，s == 0 的块 q 全为 0。
  返回 (q, s, g)。长度不是 block 的倍数时 ValueError。
- dequantize(q, scales, block, global_scale=1.0)：x̂ = q · repeat(scales, block) · global_scale。
- block_scaled_dot(qa, sa, qb, sb, block)：Σ_块 sa[i]·sb[i]·(qa_i · qb_i)，模拟 block-scaled MMA 的一个点积。
- promoted_dot(a, b, acc_bits, promote_every=None)：模拟低精度累加器。依次对 k：
  acc = round_mantissa(acc + a[k]·b[k], acc_bits)；每累加 promote_every 个 k（以及最后），
  把 acc 加进一个 float64 的 total 并清零 acc。promote_every=None 表示只在最后提升一次。返回 total。

API
    E2M1_GRID, round_e4m3(x), round_mantissa(x, bits)          given
    quantize_e2m1(x) -> np.ndarray
    mx_quantize(x, block=32, elem="e4m3") -> (q, scales)
    nvfp4_quantize(x, block=16) -> (q, s, g)
    dequantize(q, scales, block, global_scale=1.0) -> np.ndarray
    block_scaled_dot(qa, sa, qb, sb, block) -> float
    promoted_dot(a, b, acc_bits, promote_every=None) -> float
"""
from __future__ import annotations

import numpy as np

E2M1_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def round_e4m3(x):
    """就近舍入到 FP8 E4M3（round-half-even，饱和到 ±448，最小次正规数 2^-9）。"""
    x = np.asarray(x, dtype=np.float64)
    a = np.minimum(np.abs(x), 448.0)
    with np.errstate(divide="ignore"):
        e = np.floor(np.log2(np.where(a > 0, a, 1.0)))
    e = np.maximum(e, -6.0)
    q = np.exp2(e - 3)
    r = np.minimum(np.round(a / q) * q, 448.0)
    return np.where(a > 0, np.sign(x) * r, 0.0)


def round_mantissa(x, bits: int):
    """把 x 舍入到 bits 位显式尾数（round-half-even），模拟低精度累加器。"""
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    with np.errstate(divide="ignore"):
        e = np.floor(np.log2(np.where(a > 0, a, 1.0)))
    q = np.exp2(e - bits)
    r = np.round(x / q) * q
    out = np.where(a > 0, r, 0.0)
    return float(out) if out.ndim == 0 else out


def quantize_e2m1(x):
    """就近取到 E2M1 格点，中点取偶数下标，饱和到 ±6。"""
    x = np.asarray(x, dtype=np.float64)
    a = np.minimum(np.abs(x), 6.0)
    # 和每个格点的距离；平局时偏向偶数下标（尾数位 0）。格点不均匀（0.5 / 1 / 2 的间隔），
    # 所以不能用“除以步长再 round”的写法——那是常见的错误实现。
    d = np.abs(a[..., None] - E2M1_GRID)
    best = d.min(axis=-1, keepdims=True)
    cand = np.isclose(d, best, rtol=0, atol=1e-12)
    idx_even = np.where(cand & (np.arange(8) % 2 == 0), np.arange(8), 99).min(axis=-1)
    idx_any = np.where(cand, np.arange(8), 99).min(axis=-1)
    idx = np.where(idx_even < 99, idx_even, idx_any)
    return np.sign(x) * E2M1_GRID[idx]


def mx_quantize(x, block: int = 32, elem: str = "e4m3"):
    """OCP MX：每 block 个元素一个 2 的幂 scale。"""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or len(x) % block:
        raise ValueError("length must be a multiple of the block size")
    if elem == "e4m3":
        emax, rnd = 8, round_e4m3
    elif elem == "e2m1":
        emax, rnd = 2, quantize_e2m1
    else:
        raise ValueError("elem must be e4m3 or e2m1")
    xb = x.reshape(-1, block)
    amax = np.abs(xb).max(axis=1)
    with np.errstate(divide="ignore"):
        e = np.floor(np.log2(np.where(amax > 0, amax, 1.0))) - emax
    # 2 的幂 scale = 只存 8 位指数（UE8M0）。scale 让块内最大值落在元素格式最大指数的那一档，
    # 所以最大值可能被饱和截掉一点：e2m1 块 amax = 7 → floor(log2 7) − 2 = 0 → scale 1 → 7 饱和成 6，
    # 误差 1。NVFP4 的 E4M3 scale 有尾数（amax/6 可以不是 2 的幂），正是为了减小这种误差。
    e = np.clip(np.where(amax > 0, e, 0.0), -127, 127)
    scales = np.exp2(e)
    q = rnd(xb / scales[:, None])
    return q.reshape(-1), scales


def nvfp4_quantize(x, block: int = 16):
    """NVFP4：每 16 个元素一个 E4M3 scale，外加一个全张量 FP32 scale。"""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or len(x) % block:
        raise ValueError("length must be a multiple of the block size")
    amax_t = np.abs(x).max()
    # 两级 scale：g 把全张量的范围压进 “E4M3 scale × E2M1 元素” 能表示的 448·6 以内；
    # 每块的 E4M3 scale 再把块内最大值对齐到 6。块更小（16 vs 32）、scale 有 3 位尾数，
    # 所以 NVFP4 的误差通常明显小于 MXFP4。
    g = amax_t / (448.0 * 6.0) if amax_t > 0 else 1.0
    xb = x.reshape(-1, block)
    s = round_e4m3(np.abs(xb).max(axis=1) / (6.0 * g))
    safe = np.where(s > 0, s, 1.0)
    q = np.where(s[:, None] > 0, quantize_e2m1(xb / (safe[:, None] * g)), 0.0)
    return q.reshape(-1), s, g


def dequantize(q, scales, block: int, global_scale: float = 1.0):
    """x̂ = q · scale（逐块）· 全局 scale。"""
    return np.asarray(q) * np.repeat(np.asarray(scales), block) * global_scale


def block_scaled_dot(qa, sa, qb, sb, block: int) -> float:
    """block-scaled MMA 的一个点积：每块内整数/低精度点积，再乘两边的 scale。"""
    # Tensor Core 里就是这么做的：块内用 FP4/FP8 乘加，块与块之间乘 scale 后累加到 FP32。
    qa = np.asarray(qa).reshape(-1, block)
    qb = np.asarray(qb).reshape(-1, block)
    return float(np.sum(np.asarray(sa) * np.asarray(sb) * np.sum(qa * qb, axis=1)))


def promoted_dot(a, b, acc_bits: int, promote_every=None) -> float:
    """低精度累加器 + 每 promote_every 个 k 提升到 float64。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = len(a)
    step = promote_every or n
    total = 0.0
    acc = 0.0
    for k in range(n):
        # 累加器越大，新加的小量越容易被舍入吃掉（swamping）：acc=4096、尾数 13 位时，
        # 步长是 1，加 0.4 直接没了。定期把 acc 清零、加进高精度 total，acc 就一直很小。
        acc = round_mantissa(acc + a[k] * b[k], acc_bits)
        if (k + 1) % step == 0 or k == n - 1:
            total += acc
            acc = 0.0
    return total
