"""A2 · mma.sync m16n8k16 fragment layout and ldmatrix — reference solution.

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
    # 规律：行 = g 或 g+8（上下两半），列 = 2t 或 2t+8（左右两半），每个寄存器装两个相邻列。
    # 所以一个 quad（4 个相邻 lane，同一个 g）合起来正好覆盖第 g 行和第 g+8 行的全部 16 列。
    g, t = lane // 4, lane % 4
    out = []
    for r_off, c_off in ((0, 0), (8, 0), (0, 8), (8, 8)):     # a0, a1, a2, a3
        out += [(g + r_off, 2 * t + c_off), (g + r_off, 2 * t + c_off + 1)]
    return out


def b_coords(lane: int) -> list[tuple[int, int]]:
    """lane 持有的 4 个 B 元素坐标 (k, n)。"""
    # B 是“col”布局：一个寄存器里的两个元素是同一列 n、相邻两个 k。
    # 这正是 B^T（N×K、k 连续）按行存放时 ldmatrix 非转置读出来的样子。
    g, t = lane // 4, lane % 4
    return [(2 * t, g), (2 * t + 1, g), (2 * t + 8, g), (2 * t + 9, g)]


def c_coords(lane: int) -> list[tuple[int, int]]:
    """lane 持有的 4 个 C/D 元素坐标 (m, n)。"""
    # 和 A 的 a0/a1 同一套行规律：这就是 FlashAttention 能把 S = QK^T 的结果（C 布局）
    # 直接当成下一次 P·V 的 A 操作数（A 布局）的原因——行完全对得上，只差 dtype 转换。
    g, t = lane // 4, lane % 4
    return [(g, 2 * t), (g, 2 * t + 1), (g + 8, 2 * t), (g + 8, 2 * t + 1)]


def mma_m16n8k16(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """warp 级 MMA：按布局拼出 A、B、C，算 D = A@B + C，再按 C 布局分回各 lane。"""
    A = np.zeros((16, 16))
    B = np.zeros((16, 8))
    C = np.zeros((16, 8))
    for lane in range(32):
        for v, (r, k) in zip(a[lane], a_coords(lane)):
            A[r, k] = v
        for v, (k, n) in zip(b[lane], b_coords(lane)):
            B[k, n] = v
        for v, (r, n) in zip(c[lane], c_coords(lane)):
            C[r, n] = v
    D = A @ B + C
    d = np.zeros((32, 4))
    for lane in range(32):
        d[lane] = [D[r, n] for r, n in c_coords(lane)]
    return d


def ldmatrix(smem: np.ndarray, row_addrs, num: int) -> np.ndarray:
    """模拟 ldmatrix.x{num}：lane 8j..8j+7 给出矩阵 j 的 8 行地址；结果 (32, 2·num)。"""
    if num not in (1, 2, 4):
        raise ValueError("ldmatrix num must be 1, 2 or 4")
    # 关键点：“谁提供地址”和“谁拿到数据”是两回事。地址按 lane 8j+r 给出矩阵 j 的第 r 行，
    # 数据则按 mma 的 fragment 规律分发：lane l 拿到每个矩阵第 l//4 行的第 2(l%4)、2(l%4)+1 列。
    # 硬件在 warp 内部做这次“转置式”重分发，所以一条指令代替了 8 次 32-bit 的 ld.shared + shuffle。
    out = np.zeros((32, 2 * num), dtype=smem.dtype)
    for j in range(num):
        rows = [smem[row_addrs[8 * j + r]: row_addrs[8 * j + r] + 8] for r in range(8)]
        for lane in range(32):
            r, c = lane // 4, 2 * (lane % 4)
            out[lane, 2 * j] = rows[r][c]
            out[lane, 2 * j + 1] = rows[r][c + 1]
    return out


def a_tile_row_addrs(base: int, ld: int) -> list[int]:
    """ldmatrix.x4 读 16×16 A tile 时每个 lane 提供的行地址。"""
    # 寄存器 a0..a3 依次是：左上、左下、右上、右下 四个 8×8 子矩阵。
    # 矩阵 j = lane // 8：j 的低位选上下（+8 行），高位选左右（+8 列）。
    addrs = []
    for lane in range(32):
        j, r = lane // 8, lane % 8
        row = r + 8 * (j % 2)
        col = 8 * (j // 2)
        addrs.append(base + row * ld + col)
    return addrs


def b_tile_row_addrs(base: int, ld: int) -> list[int]:
    """ldmatrix.x2 读 16(K)×8(N) 的 B tile（B^T 按 N×K 存）时每个 lane 提供的行地址。"""
    # 每一行是一个 n、8 个连续 k。矩阵 0 = k 0..7，矩阵 1 = k 8..15。
    # 若 B 是 K×N 存放（n 连续），就得用 ldmatrix.trans，或者改用 TN 布局——
    # 这就是 CUTLASS 里 “TN GEMM” 最常见、最快的原因。
    addrs = []
    for lane in range(32):
        if lane < 16:
            j, r = lane // 8, lane % 8
            addrs.append(base + r * ld + 8 * j)
        else:
            addrs.append(base)
    return addrs
