"""A2 · mma.sync m16n8k16 fragment layout and ldmatrix — your implementation (run ./check.sh a2).

从 Volta 开始，矩阵乘是“一个 warp 一条指令”：32 个线程各自拿着矩阵的几个元素（fragment），
一起执行 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`。写 Ampere 风格 kernel 的核心难点，
就是让每个线程寄存器里放的恰好是指令要的那几个元素。这道题用 numpy 把 PTX 文档里的布局写出来，
并模拟 `ldmatrix` 怎么一条指令把 SMEM 里的 tile 直接读成这个布局。

记号：lane ∈ [0, 32)，g = lane // 4（groupID），t = lane % 4（threadID_in_group）。

- a_coords(lane): A（16×16，row-major 语义）的 8 个元素，按寄存器顺序 a0..a3、每个寄存器低半在前：
    a0 = (g, 2t), (g, 2t+1)      a1 = (g+8, 2t), (g+8, 2t+1)
    a2 = (g, 2t+8), (g, 2t+9)    a3 = (g+8, 2t+8), (g+8, 2t+9)
  返回 8 个 (row, col)。
- b_coords(lane): B（16×8，K×N）的 4 个元素：b0 = (2t, g), (2t+1, g)；b1 = (2t+8, g), (2t+9, g)。返回 (k, n)。
- c_coords(lane): C/D（16×8 FP32）的 4 个元素：(g, 2t), (g, 2t+1), (g+8, 2t), (g+8, 2t+1)。返回 (m, n)。
- mma_m16n8k16(a, b, c): a 形状 (32, 8)、b (32, 4)、c (32, 4)，第 i 行是 lane i 按上面顺序持有的元素。
  返回 d (32, 4)：D = A @ B + C，按 c_coords 分回各 lane。用 float64 计算即可。
- ldmatrix(smem, row_addrs, num): 模拟 `ldmatrix.sync.aligned.m8n8.x{num}.shared.b16`（num ∈ {1,2,4}）。
  smem 是 1D 数组（元素 = b16），row_addrs 是 32 个元素偏移：第 j 个 8×8 矩阵的 8 行地址由
  lane 8j..8j+7 提供（num 之外的 lane 地址被忽略）。每行 8 个连续元素。
  结果 (32, 2·num)：lane l 的第 j 个寄存器（列 2j、2j+1）= 矩阵 j 的第 l//4 行、第 2·(l%4) 和 +1 列。
  num 不是 1/2/4 时抛 ValueError。
- a_tile_row_addrs(base, ld): 用 ldmatrix.x4 读一个从 base 开始、行跨度 ld 的 16×16 A tile 时，
  32 个 lane 各自该提供的行地址，使得结果恰好等于 a_coords 的布局。
- b_tile_row_addrs(base, ld): B 以 N×K（每行一个 n、k 连续，即 “TN” 里的 B^T）存放、从 base 开始、
  行跨度 ld 时，ldmatrix.x2 读一个 16(K)×8(N) 的 B tile 所需的地址（lane 0..15 有效，其余填 base 即可），
  使结果等于 b_coords 的布局。

API
    a_coords(lane) -> list[tuple[int, int]]
    b_coords(lane) -> list[tuple[int, int]]
    c_coords(lane) -> list[tuple[int, int]]
    mma_m16n8k16(a, b, c) -> np.ndarray
    ldmatrix(smem, row_addrs, num) -> np.ndarray
    a_tile_row_addrs(base, ld) -> list[int]
    b_tile_row_addrs(base, ld) -> list[int]
"""
from __future__ import annotations

import numpy as np


def a_coords(lane: int) -> list[tuple[int, int]]:
    """lane 持有的 8 个 A 元素坐标（寄存器 a0..a3，每个两个元素）。"""
    raise NotImplementedError("a_coords")


def b_coords(lane: int) -> list[tuple[int, int]]:
    """lane 持有的 4 个 B 元素坐标 (k, n)。"""
    raise NotImplementedError("b_coords")


def c_coords(lane: int) -> list[tuple[int, int]]:
    """lane 持有的 4 个 C/D 元素坐标 (m, n)。"""
    raise NotImplementedError("c_coords")


def mma_m16n8k16(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """warp 级 MMA：按布局拼出 A、B、C，算 D = A@B + C，再按 C 布局分回各 lane。"""
    raise NotImplementedError("mma_m16n8k16")


def ldmatrix(smem: np.ndarray, row_addrs, num: int) -> np.ndarray:
    """模拟 ldmatrix.x{num}：lane 8j..8j+7 给出矩阵 j 的 8 行地址；结果 (32, 2·num)。"""
    raise NotImplementedError("ldmatrix")


def a_tile_row_addrs(base: int, ld: int) -> list[int]:
    """ldmatrix.x4 读 16×16 A tile 时每个 lane 提供的行地址。"""
    raise NotImplementedError("a_tile_row_addrs")


def b_tile_row_addrs(base: int, ld: int) -> list[int]:
    """ldmatrix.x2 读 16(K)×8(N) 的 B tile（B^T 按 N×K 存）时每个 lane 提供的行地址。"""
    raise NotImplementedError("b_tile_row_addrs")
