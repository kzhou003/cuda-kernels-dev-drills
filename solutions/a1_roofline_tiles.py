"""A1 · Roofline, tile arithmetic intensity and the register / SMEM budget — reference solution.

每一代 GPU 的 Tensor Core 都比上一代快一倍，但 HBM 和 SMEM 带宽涨得慢得多。这道题把
“需要多大的 tile 才喂得饱 Tensor Core”算清楚：CUTLASS / cuBLAS 选 tile 形状、FA 选块大小时
心里算的就是这几笔账。

- machine_balance(gpu): 峰值 Tensor FLOP/s ÷ HBM 字节/s（FLOP/byte）。
- smem_balance(gpu): 每 SM 每时钟 Tensor FLOP ÷ 每 SM 每时钟 SMEM 字节（gpu.smem_bytes_per_clk）。
- tile_intensity(M, N, elem_bytes=2): 一个 M×N 输出 tile 每走一步 k 的强度
  = 2·M·N FLOP ÷ ((M + N)·elem_bytes) 字节。
- gemm_intensity(M, N, K, elem_bytes=2, out_bytes=2): 整个 GEMM 在“每个元素只读写一次”的理想情况下
  = 2·M·N·K ÷ (elem_bytes·(M·K + K·N) + out_bytes·M·N)。
- attainable_tflops(gpu, intensity): roofline = min(峰值, intensity × HBM 带宽)，单位 TFLOP/s。
- smallest_feeding_tile(gpu, candidates, elem_bytes=2): 在 candidates（(M, N) 列表）里按面积 M·N
  从小到大（面积相同按 M 小的优先）找第一个 tile_intensity ≥ smem_balance(gpu) 的；没有返回 None。
- accumulator_regs_per_thread(M, N, threads, acc_bytes=4): FP32 累加器平均到每个线程要几个 32-bit 寄存器
  = ceil(M·N·acc_bytes / 4 / threads)。
- ctas_per_sm(gpu, threads_per_cta, regs_per_thread, smem_per_cta): 一个 SM 同时能驻留几个 CTA，
  取下面四个上限的最小值：
    寄存器：每个 warp 的寄存器按 regs_per_thread 先向上取整到 8 的倍数再 ×32，
            CTA 的 warp 数 = ceil(threads/32)；上限 = gpu.regs_per_sm // (warps × 每 warp 寄存器)；
    SMEM：smem_per_cta > gpu.smem_per_cta_max 时返回 0；smem_per_cta == 0 时不受限；
          否则上限 = gpu.smem_per_sm // smem_per_cta；
    线程：gpu.max_threads_per_sm // threads_per_cta；
    CTA 数：gpu.max_ctas_per_sm。
  regs_per_thread > 255 时返回 0（编译器会 spill，硬件不允许）。

API
    GPU, GPUS                                              given：规格表
    machine_balance(gpu) -> float
    smem_balance(gpu) -> float
    tile_intensity(M, N, elem_bytes=2) -> float
    gemm_intensity(M, N, K, elem_bytes=2, out_bytes=2) -> float
    attainable_tflops(gpu, intensity) -> float
    smallest_feeding_tile(gpu, candidates, elem_bytes=2) -> tuple[int, int] | None
    accumulator_regs_per_thread(M, N, threads, acc_bytes=4) -> int
    ctas_per_sm(gpu, threads_per_cta, regs_per_thread, smem_per_cta) -> int
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GPU:
    """一代 GPU 的关键规格（SXM / 公版，dense BF16 Tensor Core，FP32 累加）。

    tc_flops_per_clk_per_sm 由“峰值 ÷ (SM 数 × boost 时钟)”反推，取整到 2 的幂；
    smem_bytes_per_clk 是每 SM 每时钟的 SMEM 带宽（这几代都是 128 B）。
    """
    name: str
    sm: str
    num_sms: int
    clock_ghz: float
    tc_flops_per_clk_per_sm: int
    hbm_gbps: float
    smem_bytes_per_clk: int = 128
    regs_per_sm: int = 65536
    smem_per_sm: int = 0          # bytes
    smem_per_cta_max: int = 0     # bytes
    max_threads_per_sm: int = 2048
    max_ctas_per_sm: int = 32

    @property
    def peak_tflops(self) -> float:
        return self.num_sms * self.clock_ghz * 1e9 * self.tc_flops_per_clk_per_sm / 1e12


GPUS = {
    "V100": GPU("V100", "sm70", 80, 1.53, 1024, 900, smem_per_sm=96 * 1024, smem_per_cta_max=96 * 1024),
    "A100": GPU("A100", "sm80", 108, 1.41, 2048, 2039, smem_per_sm=164 * 1024, smem_per_cta_max=163 * 1024),
    "H100": GPU("H100", "sm90a", 132, 1.83, 4096, 3350, smem_per_sm=228 * 1024, smem_per_cta_max=227 * 1024),
    "B200": GPU("B200", "sm100a", 148, 1.86, 8192, 8000, smem_per_sm=228 * 1024, smem_per_cta_max=227 * 1024),
    "RTX5090": GPU("RTX5090", "sm120", 170, 2.41, 512, 1792, smem_per_sm=100 * 1024, smem_per_cta_max=99 * 1024,
                   max_threads_per_sm=1536),
}


def machine_balance(gpu: GPU) -> float:
    """峰值 Tensor FLOP/s ÷ HBM 字节/s。kernel 的强度低于它就是 memory-bound。"""
    # H100: 989 TF / 3.35 TB/s ≈ 295 FLOP/byte。这是“整卡”的账：数据从 HBM 来一次，
    # 在片上要被用 295 次才不浪费 Tensor Core。
    return gpu.peak_tflops * 1e12 / (gpu.hbm_gbps * 1e9)


def smem_balance(gpu: GPU) -> float:
    """每 SM 每时钟 Tensor FLOP ÷ 每 SM 每时钟 SMEM 字节。"""
    # 这是“每个 SM”的账：操作数从 SMEM 读进 Tensor Core，每个字节要被用多少次。
    # V100 8 → A100 16 → H100 32 → B200 64，每代翻倍，因为 SMEM 带宽一直是 128 B/clk。
    # RTX 5090 只有 4：GeForce 的 FP32 累加 Tensor Core 吞吐只有数据中心卡的几分之一，
    # 所以在 5090 上小 tile 也能喂饱它——这也是为什么 mma.sync 在 5090 上依然够用。
    return gpu.tc_flops_per_clk_per_sm / gpu.smem_bytes_per_clk


def tile_intensity(M: int, N: int, elem_bytes: int = 2) -> float:
    """M×N 输出 tile 每走一步 k：2·M·N FLOP ÷ (M + N)·elem_bytes 字节。"""
    # 一步 k 读 A 的一列（M 个元素）和 B 的一行（N 个元素），做 M·N 次乘加。
    # 面积 / 周长：tile 越接近方形、越大，强度越高。
    return 2 * M * N / ((M + N) * elem_bytes)


def gemm_intensity(M: int, N: int, K: int, elem_bytes: int = 2, out_bytes: int = 2) -> float:
    """整个 GEMM 的理想强度（A、B 各读一次，C 写一次）。"""
    # 理想情况，现实中 A/B 会被不同 CTA 重复读，L2 命中率决定离理想有多远（C3 的 raster swizzle）。
    flops = 2 * M * N * K
    bytes_ = elem_bytes * (M * K + K * N) + out_bytes * M * N
    return flops / bytes_


def attainable_tflops(gpu: GPU, intensity: float) -> float:
    """roofline：min(峰值, 强度 × HBM 带宽)，TFLOP/s。"""
    return min(gpu.peak_tflops, intensity * gpu.hbm_gbps * 1e9 / 1e12)


def smallest_feeding_tile(gpu: GPU, candidates, elem_bytes: int = 2):
    """按面积从小到大找第一个强度 ≥ smem_balance 的 tile；都不行返回 None。"""
    # 为什么要“最小”的那个：tile 越大，累加器越多（寄存器 / TMEM 压力），
    # 而且 M、N 不整除时浪费越多（尾 tile），所以只要喂得饱就不再加大。
    need = smem_balance(gpu)
    for M, N in sorted(candidates, key=lambda t: (t[0] * t[1], t[0])):
        if tile_intensity(M, N, elem_bytes) >= need:
            return (M, N)
    return None


def accumulator_regs_per_thread(M: int, N: int, threads: int, acc_bytes: int = 4) -> int:
    """FP32 累加器平均到每个线程的 32-bit 寄存器数（向上取整）。"""
    # 128×256 的 FP32 累加器 = 128 KB = 半个寄存器堆。分给一个 warpgroup（128 线程）
    # 就是每线程 256 个寄存器——超过 255 的上限，所以 Hopper 要两个 consumer warpgroup 分，
    # Blackwell 干脆把累加器搬进 TMEM。
    return math.ceil(M * N * acc_bytes / 4 / threads)


def ctas_per_sm(gpu: GPU, threads_per_cta: int, regs_per_thread: int, smem_per_cta: int) -> int:
    """一个 SM 能同时驻留的 CTA 数（寄存器 / SMEM / 线程 / CTA 四个上限取最小）。"""
    if regs_per_thread > 255:
        return 0
    if smem_per_cta > gpu.smem_per_cta_max:
        return 0
    warps = math.ceil(threads_per_cta / 32)
    # 寄存器按 warp 分配，粒度 256 个（= 每线程 8 个）。所以 regs=33 实际按 40 算——
    # 这是“我只多用了 1 个寄存器，occupancy 怎么掉了”的常见原因。
    regs_per_warp = math.ceil(regs_per_thread / 8) * 8 * 32
    by_regs = gpu.regs_per_sm // (warps * regs_per_warp) if regs_per_warp else gpu.max_ctas_per_sm
    by_smem = gpu.smem_per_sm // smem_per_cta if smem_per_cta else gpu.max_ctas_per_sm
    by_threads = gpu.max_threads_per_sm // threads_per_cta
    return min(by_regs, by_smem, by_threads, gpu.max_ctas_per_sm)
