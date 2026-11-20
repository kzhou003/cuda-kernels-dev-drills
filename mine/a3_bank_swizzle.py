"""A3 · Shared-memory bank conflicts and the CuTe Swizzle<B, M, S> — your implementation (run ./check.sh a3).

SMEM 分成 32 个 bank，每个 bank 4 字节宽，一个时钟每个 bank 只能服务一个 4 字节字。
同一 warp 的访问落在同一 bank 的不同字上，就要拆成多次（wavefront）。Tensor Core kernel 里
最常见的冲突：tile 按行存（每行 128 B），ldmatrix 却按“列方向”一次读 8 行的同一个 16 B 块，
8 行全落在同一组 bank 上。解决办法有两种：padding（每行多留几个元素）和 swizzle（把列块编号
和行号做 XOR）。CuTe / CUTLASS / TMA 都用 swizzle，因为它不浪费 SMEM，而且 TMA 能直接按它写。

- wavefronts_32bit(byte_addrs): 一个 warp 的 32 个 4 字节访问需要几个 wavefront。
  bank = (addr // 4) % 32；同一 bank 上**不同**的字（addr // 4 不同）各需一次，同一个字是广播，不算冲突。
  结果 = 各 bank 不同字数的最大值（至少 1；空列表返回 0）。
- ldmatrix_wavefronts(row_byte_addrs): ldmatrix 每个 8×8 矩阵读 8 行、每行 16 B（一个 128 B 相位）。
  row_byte_addrs 按 8 个一组（一组一个矩阵）。每组里 16 B 块所在的“槽” = (addr // 16) % 8，
  同一槽上不同的 16 B 块（addr // 16 不同）各需一次。结果 = 各组最大值之和。
  地址不是 16 的倍数、或长度不是 8 的倍数时抛 ValueError。
- swizzle(offset, B, M, S): CuTe Swizzle<B, M, S>（S > 0 的情形）：
  取 offset 的第 [M+S, M+S+B) 位，右移 S 位后 XOR 到第 [M, M+B) 位上；其余位不变。
- swizzled_offset(row, col, ld, B, M, S) = swizzle(row·ld + col, B, M, S)（元素偏移）。
- padded_offset(row, col, ld, pad) = row·(ld + pad) + col。

API
    wavefronts_32bit(byte_addrs) -> int
    ldmatrix_wavefronts(row_byte_addrs) -> int
    swizzle(offset, B, M, S) -> int
    swizzled_offset(row, col, ld, B, M, S) -> int
    padded_offset(row, col, ld, pad) -> int
"""
from __future__ import annotations

from collections import defaultdict


def wavefronts_32bit(byte_addrs) -> int:
    """32 个 4 字节访问需要的 wavefront 数（广播不算冲突）。"""
    raise NotImplementedError("wavefronts_32bit")


def ldmatrix_wavefronts(row_byte_addrs) -> int:
    """ldmatrix 每 8 行一组（一个 128 B 相位），同槽不同 16 B 块要串行。"""
    raise NotImplementedError("ldmatrix_wavefronts")


def swizzle(offset: int, B: int, M: int, S: int) -> int:
    """CuTe Swizzle<B,M,S>：把 [M+S, M+S+B) 位 XOR 到 [M, M+B) 位。"""
    raise NotImplementedError("swizzle")


def swizzled_offset(row: int, col: int, ld: int, B: int, M: int, S: int) -> int:
    """先按行主序算偏移，再 swizzle。"""
    raise NotImplementedError("swizzled_offset")


def padded_offset(row: int, col: int, ld: int, pad: int) -> int:
    """每行多留 pad 个元素。"""
    raise NotImplementedError("padded_offset")
