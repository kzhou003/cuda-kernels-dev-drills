"""B3 · TMA box copy: out-of-bounds fill, 128B swizzle, transaction bytes and cluster multicast — reference solution.

Hopper 的 TMA（Tensor Memory Accelerator）把“128 个线程各算地址、各发 16 B”变成“一个线程发一条
指令搬整个 box”。描述符（CUtensorMap）里写好全局张量的形状、跨度、box 大小和 swizzle 方式，
硬件负责：越界填零、按 swizzle 写进 SMEM、写完按字节数给 mbarrier 记账（complete_tx）。
cluster 里的 multicast 再让一次 L2 读取写进多个 CTA 的 SMEM。这道题在 numpy 上复刻这些行为。

- tma_load_2d(gmem, row0, col0, box_rows, box_cols, swizzle_bytes=0, elem_bytes=2) -> 1D np.ndarray：
  从 2D 数组 gmem 取以 (row0, col0) 为左上角的 box（坐标可以为负或越界），越界元素填 0；
  按行主序放进长 box_rows·box_cols 的 SMEM 数组。逻辑 (r, c) 的字节偏移 o = (r·box_cols + c)·elem_bytes，
  swizzle_bytes 为 32 / 64 / 128 时 o 先经过 swizzle_byte_offset，再除以 elem_bytes 得到下标。
  swizzle_bytes 不在 {0, 32, 64, 128}，或开了 swizzle 但 box_cols·elem_bytes > swizzle_bytes，抛 ValueError。
- swizzle_byte_offset(o, swizzle_bytes)：CU_TENSOR_MAP_SWIZZLE_{32B,64B,128B} = 以 16 B 为单位的 Swizzle<B,4,3>，
  B = 1 / 2 / 3：把 o 的第 [7, 7+B) 位右移 3 位后 XOR 到第 [4, 4+B) 位。swizzle_bytes = 0 时原样返回。
- smem_read(smem, r, c, box_cols, swizzle_bytes=0, elem_bytes=2)：consumer 按同一规则读逻辑 (r, c)。
- tx_bytes(box_rows, box_cols, elem_bytes=2)：这次拷贝给 mbarrier 记的字节数 = 整个 box 的字节数
  （越界填零的部分也算）。
- tile_boxes(M, N, box_rows, box_cols)：覆盖 M×N 矩阵所需的全部 box 左上角，按行优先列出（尾块会越界）。
- multicast_masks(cm, cn)：cm×cn 的 cluster，CTA (x, y) 的 rank = x + y·cm，负责输出 tile (x, y)。
  A tile 只依赖 x，B tile 只依赖 y。返回 {rank: (a_mask, b_mask)}：a_mask 是所有同 x 的 CTA 的 rank 位，
  b_mask 是所有同 y 的 CTA 的 rank 位（都包含自己）。
- l2_bytes_per_kstep(m_tiles, n_tiles, a_tile_bytes, b_tile_bytes, cm=1, cn=1)：每走一步 k，整个 grid
  从 L2 读多少字节。不用 multicast（cm = cn = 1）时每个 CTA 各读自己的 A、B；用了以后每份 A 只读一次、
  分给 cn 个 CTA，每份 B 只读一次、分给 cm 个 CTA。m_tiles 不能被 cm 整除或 n_tiles 不能被 cn 整除时抛 ValueError。

API
    tma_load_2d(gmem, row0, col0, box_rows, box_cols, swizzle_bytes=0, elem_bytes=2) -> np.ndarray
    swizzle_byte_offset(o, swizzle_bytes) -> int
    smem_read(smem, r, c, box_cols, swizzle_bytes=0, elem_bytes=2)
    tx_bytes(box_rows, box_cols, elem_bytes=2) -> int
    tile_boxes(M, N, box_rows, box_cols) -> list[tuple[int, int]]
    multicast_masks(cm, cn) -> dict[int, tuple[int, int]]
    l2_bytes_per_kstep(m_tiles, n_tiles, a_tile_bytes, b_tile_bytes, cm=1, cn=1) -> float
"""
from __future__ import annotations

import math

import numpy as np

_SWZ_B = {32: 1, 64: 2, 128: 3}


def swizzle_byte_offset(o: int, swizzle_bytes: int) -> int:
    """TMA 的 32/64/128B swizzle：16 B 块号 ^= (o >> 7) 的低 B 位。"""
    if swizzle_bytes == 0:
        return o
    # 128B：一个 1024 B 的“原子”= 8 行 × 128 B。第 r 行（o 的第 7..9 位）里的第 c 个 16 B 块
    # 放到第 c ^ (r % 8) 个位置。和 CuTe 的 Swizzle<3,4,3>（字节单位）完全一样，
    # 所以 CUTLASS 用 TMA 写、用 ldmatrix / wgmma 描述符读时，两边约定同一个函数就对上了。
    B = _SWZ_B[swizzle_bytes]
    mask = ((1 << B) - 1) << 7
    return o ^ ((o & mask) >> 3)


def _check(box_cols: int, swizzle_bytes: int, elem_bytes: int) -> None:
    if swizzle_bytes not in (0, 32, 64, 128):
        raise ValueError("swizzle must be 0, 32, 64 or 128 bytes")
    if swizzle_bytes and box_cols * elem_bytes > swizzle_bytes:
        # 真实 API（cuTensorMapEncodeTiled）同样拒绝：内维 box 字节数不能超过 swizzle 跨度。
        # 所以 128B swizzle + bf16 的 box 内维最多 64 个元素，K 方向更宽的 tile 要拆成多个 box。
        raise ValueError("inner box dimension exceeds the swizzle span")


def tma_load_2d(gmem: np.ndarray, row0: int, col0: int, box_rows: int, box_cols: int,
                swizzle_bytes: int = 0, elem_bytes: int = 2) -> np.ndarray:
    """一条 TMA 指令：越界填零 + swizzle 写入 SMEM。"""
    _check(box_cols, swizzle_bytes, elem_bytes)
    H, W = gmem.shape
    smem = np.zeros(box_rows * box_cols, dtype=gmem.dtype)
    for r in range(box_rows):
        for c in range(box_cols):
            gr, gc = row0 + r, col0 + c
            # 越界由硬件填零：kernel 里不再需要 `if (row < M)` 的谓词，尾 tile 和整 tile 走同一条路径。
            v = gmem[gr, gc] if 0 <= gr < H and 0 <= gc < W else 0
            o = swizzle_byte_offset((r * box_cols + c) * elem_bytes, swizzle_bytes)
            smem[o // elem_bytes] = v
    return smem


def smem_read(smem: np.ndarray, r: int, c: int, box_cols: int, swizzle_bytes: int = 0, elem_bytes: int = 2):
    """按同一 swizzle 规则读逻辑坐标 (r, c)。"""
    o = swizzle_byte_offset((r * box_cols + c) * elem_bytes, swizzle_bytes)
    return smem[o // elem_bytes]


def tx_bytes(box_rows: int, box_cols: int, elem_bytes: int = 2) -> int:
    """mbarrier 要等的字节数：整个 box，包括越界填零的部分。"""
    # 常见 bug：尾 tile 按“实际有效元素”算 expect_tx，结果 barrier 永远等不到（或提前完成）。
    # TMA 总是按 box 大小记账。
    return box_rows * box_cols * elem_bytes


def tile_boxes(M: int, N: int, box_rows: int, box_cols: int) -> list[tuple[int, int]]:
    """覆盖 M×N 需要的 box 左上角（行优先）。"""
    return [(r, c) for r in range(0, math.ceil(M / box_rows) * box_rows, box_rows)
            for c in range(0, math.ceil(N / box_cols) * box_cols, box_cols)]


def multicast_masks(cm: int, cn: int) -> dict[int, tuple[int, int]]:
    """cluster 里每个 CTA 的 A / B multicast 掩码。"""
    # 同一列（同 x）的 CTA 算的是同一段 M，需要同一个 A tile；同一行（同 y）需要同一个 B tile。
    # CUTLASS 里每个 CTA 只发自己那一份切片，但用 mask 把它写进所有需要它的 CTA 的 SMEM，
    # 所以 L2 → SM 的流量按 cluster 维度成倍减少。
    out = {}
    for y in range(cn):
        for x in range(cm):
            a = sum(1 << (x + yy * cm) for yy in range(cn))
            b = sum(1 << (xx + y * cm) for xx in range(cm))
            out[x + y * cm] = (a, b)
    return out


def l2_bytes_per_kstep(m_tiles: int, n_tiles: int, a_tile_bytes: int, b_tile_bytes: int,
                       cm: int = 1, cn: int = 1) -> float:
    """每步 k 整个 grid 从 L2 读的字节数（考虑 multicast）。"""
    if m_tiles % cm or n_tiles % cn:
        raise ValueError("tile grid must be divisible by the cluster shape")
    return m_tiles * n_tiles * (a_tile_bytes / cn + b_tile_bytes / cm)
