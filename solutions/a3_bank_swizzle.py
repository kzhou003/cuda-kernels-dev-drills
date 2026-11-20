"""A3 · Shared-memory bank conflicts and the CuTe Swizzle<B, M, S> — reference solution.

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
    # 只数“不同的字”：32 个线程读同一个地址是一次广播，1 个 wavefront。
    # 常见错误是数“落在同一 bank 的线程数”，那样广播会被误判成 32 路冲突。
    if not byte_addrs:
        return 0
    words = defaultdict(set)
    for a in byte_addrs:
        w = a // 4
        words[w % 32].add(w)
    return max(len(s) for s in words.values())


def ldmatrix_wavefronts(row_byte_addrs) -> int:
    """ldmatrix 每 8 行一组（一个 128 B 相位），同槽不同 16 B 块要串行。"""
    addrs = list(row_byte_addrs)
    if len(addrs) % 8 or any(a % 16 for a in addrs):
        raise ValueError("ldmatrix rows are 16-byte aligned, 8 per matrix")
    total = 0
    for i in range(0, len(addrs), 8):
        # 8 行 × 16 B = 128 B = 正好 32 个 bank 一圈。8 个 16 B 块落在 8 个不同的“槽”
        # （每槽 4 个 bank）时一次读完；按行存放、每行 128 B 的 tile，8 行同一列块全在同一槽 → 8 次。
        slots = defaultdict(set)
        for a in addrs[i:i + 8]:
            chunk = a // 16
            slots[chunk % 8].add(chunk)
        total += max(len(s) for s in slots.values())
    return total


def swizzle(offset: int, B: int, M: int, S: int) -> int:
    """CuTe Swizzle<B,M,S>：把 [M+S, M+S+B) 位 XOR 到 [M, M+B) 位。"""
    # 读法（以 bf16、Swizzle<3,3,3> 为例，单位是元素）：
    #   M=3：低 3 位（8 个元素 = 16 B）原样保留——ldmatrix / cp.async 的 16 B 向量不被拆开；
    #   B=3：有 8 个 16 B 块要打散；
    #   S=3：用“往上 3 位”的那几位做钥匙，即 offset 的第 6..8 位 = 行号的低 3 位（每行 64 个元素）。
    # 结果：第 r 行的第 c 个 16 B 块被放到第 c ^ (r % 8) 个位置。同一“列块”的 8 行分到 8 个不同槽。
    # XOR 是自己的逆，所以写和读用同一个函数；它在每 2^(B+M+S) 个元素内是一个置换，不浪费空间。
    mask = ((1 << B) - 1) << (M + S)
    return offset ^ ((offset & mask) >> S)


def swizzled_offset(row: int, col: int, ld: int, B: int, M: int, S: int) -> int:
    """先按行主序算偏移，再 swizzle。"""
    return swizzle(row * ld + col, B, M, S)


def padded_offset(row: int, col: int, ld: int, pad: int) -> int:
    """每行多留 pad 个元素。"""
    # padding 的代价：多占 SMEM（bf16、每行 64 元素加 8 → 12.5%），而且 TMA 没法按 padding 写入，
    # 所以 Hopper 以后基本只用 swizzle。Ampere 的手写 kernel、FA2 的部分路径还在用 padding。
    return row * (ld + pad) + col
