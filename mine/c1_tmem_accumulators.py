"""C1 · Blackwell TMEM: column allocation, double-buffered accumulators and 2-CTA MMA — your implementation (run ./check.sh c1).

B200 每个 SM 每时钟 8192 FLOP：128×256 的 FP32 累加器已经占了半个寄存器堆，寄存器再也装不下。
Blackwell 的回答是 Tensor Memory（TMEM）：每个 SM 256 KB = 128 行（lane）× 512 列 × 32 bit，
专门放累加器。`tcgen05.mma` 由一个线程发起、把结果写进 TMEM；epilogue warp 再用 `tcgen05.ld`
读出来。TMEM 要像 malloc 一样按“列”申请（`tcgen05.alloc`，2 的幂、32..512 列），
而 512 列刚好放下两个 128×256 的累加器——MMA warp 写一个、epilogue warp 读另一个。

- TmemAllocator(capacity=512)：
  - alloc(ncols) -> int | None：ncols 必须是 [32, capacity] 内 2 的幂，否则 ValueError。
    本题的放置规则（简化模型）：从低到高找第一个“起点是 ncols 的倍数、且 [起点, 起点+ncols) 全空”的位置，
    返回起始列；没有返回 None。
  - free(start, ncols)：必须恰好对应一次尚未释放的 alloc，否则 ValueError。
  - used：已分配列数。
- acc_columns(tile_n, acc_bits=32)：一个（每 CTA）M = 64 或 128 行的累加器占多少列 = tile_n·acc_bits / 32
  （每个 lane 一行；M = 64 也按 tile_n 列占，只是只用一半 lane）。
- max_acc_buffers(tile_n, capacity=512, acc_bits=32)：最多放几个累加器 = capacity // acc_columns。
- acc_pipeline(n_tiles, ML, EP, buffers) -> float：MMA warp 依次算 tile 0..n−1，tile i 写累加器 i % buffers；
  tile i 的 mainloop 开始 = max(tile i−1 的 mainloop 结束, tile i−buffers 的 epilogue 结束)；
  epilogue 只有一组 warp，按顺序做：tile i 的 epilogue 开始 = max(tile i 的 mainloop 结束,
  tile i−1 的 epilogue 结束)，花 EP。返回最后一个 epilogue 的结束时刻（n_tiles = 0 时为 0）。
  buffers < 1 抛 ValueError。
- per_sm_intensity(tile_m, tile_n, cta_pair=False)：每个 SM 每步 k 的 FLOP / SMEM 字节（BF16）。
  单 CTA：tile_m·tile_n / (tile_m + tile_n)。cta_pair=True 时 tile_m×tile_n 是两个 SM 合作的 tile：
  每个 SM 算 tile_m/2 行、只装一半的 B（tile_n/2 列），强度 = (tile_m/2)·tile_n / (tile_m/2 + tile_n/2)。

API
    TmemAllocator(capacity=512)  .alloc(ncols) -> int | None  .free(start, ncols)  .used
    acc_columns(tile_n, acc_bits=32) -> int
    max_acc_buffers(tile_n, capacity=512, acc_bits=32) -> int
    acc_pipeline(n_tiles, ML, EP, buffers) -> float
    per_sm_intensity(tile_m, tile_n, cta_pair=False) -> float
"""
from __future__ import annotations


class TmemAllocator:
    def __init__(self, capacity: int = 512):
        """capacity 列的 TMEM（B200 是 512；CUTLASS 里 Rubin / Sm107 是 576）。"""
        raise NotImplementedError("TmemAllocator.__init__")

    def alloc(self, ncols: int):
        """申请 ncols 列；放不下返回 None。"""
        raise NotImplementedError("TmemAllocator.alloc")

    def free(self, start: int, ncols: int) -> None:
        """释放一次分配。"""
        raise NotImplementedError("TmemAllocator.free")

    @property
    def used(self) -> int:
        raise NotImplementedError("TmemAllocator.used")


def acc_columns(tile_n: int, acc_bits: int = 32) -> int:
    """累加器占的 TMEM 列数。"""
    raise NotImplementedError("acc_columns")


def max_acc_buffers(tile_n: int, capacity: int = 512, acc_bits: int = 32) -> int:
    """TMEM 里最多能放几个这样的累加器。"""
    raise NotImplementedError("max_acc_buffers")


def acc_pipeline(n_tiles: int, ML: float, EP: float, buffers: int) -> float:
    """MMA warp 和 epilogue warp 通过 buffers 个 TMEM 累加器流水。"""
    raise NotImplementedError("acc_pipeline")


def per_sm_intensity(tile_m: int, tile_n: int, cta_pair: bool = False) -> float:
    """每 SM 每步 k 的 FLOP / SMEM 字节（BF16）。"""
    raise NotImplementedError("per_sm_intensity")
