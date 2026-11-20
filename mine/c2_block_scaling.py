"""C2 · Block-scaled low precision: MXFP8 / MXFP4 / NVFP4 and DeepGEMM's FP8 promotion — your implementation (run ./check.sh c2).

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
    raise NotImplementedError("quantize_e2m1")


def mx_quantize(x, block: int = 32, elem: str = "e4m3"):
    """OCP MX：每 block 个元素一个 2 的幂 scale。"""
    raise NotImplementedError("mx_quantize")


def nvfp4_quantize(x, block: int = 16):
    """NVFP4：每 16 个元素一个 E4M3 scale，外加一个全张量 FP32 scale。"""
    raise NotImplementedError("nvfp4_quantize")


def dequantize(q, scales, block: int, global_scale: float = 1.0):
    """x̂ = q · scale（逐块）· 全局 scale。"""
    raise NotImplementedError("dequantize")


def block_scaled_dot(qa, sa, qb, sb, block: int) -> float:
    """block-scaled MMA 的一个点积：每块内整数/低精度点积，再乘两边的 scale。"""
    raise NotImplementedError("block_scaled_dot")


def promoted_dot(a, b, acc_bits: int, promote_every=None) -> float:
    """低精度累加器 + 每 promote_every 个 k 提升到 float64。"""
    raise NotImplementedError("promoted_dot")
