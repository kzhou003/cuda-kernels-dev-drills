lede: TMA 把“128 个线程各算地址、各搬 16 字节”变成“一个线程发一条指令搬整个 box”：越界填零、swizzle 写入、按字节给 mbarrier 记账都由硬件做，cluster multicast 还能一次 L2 读取写进多个 CTA 的 SMEM。
# B3 · TMA box 拷贝：越界填零、128B swizzle、cluster multicast

## 题面速览

- **在系统中的位置**：CuTe 的 [[cutlass:include/cute/arch/copy_sm90_tma.hpp@SM90_TMA_LOAD_2D|SM90_TMA_LOAD_2D]] 发 `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes`；multicast 版本是 [[cutlass:include/cute/arch/copy_sm90_tma.hpp@SM90_TMA_LOAD_MULTICAST_2D|SM90_TMA_LOAD_MULTICAST_2D]]（多一个 `ctaMask`）。描述符由 host 端 `cuTensorMapEncodeTiled` 生成（全局形状、跨度、box 大小、swizzle 模式）。RTX 5090（sm120）有 TMA 但目标只能是 `shared::cta`，**没有 multicast**。
- **你要实现**：`tma_load_2d`、`swizzle_byte_offset`、`smem_read`、`tx_bytes`、`tile_boxes`、`multicast_masks`、`l2_bytes_per_kstep`。
- **难点在这几条规则**：
  1. 坐标可以越界、可以为负；越界元素填 0，kernel 里不需要任何边界判断。
  2. 128B / 64B / 32B swizzle = 以 16 B 为单位的 `Swizzle<3/2/1, 4, 3>`（字节地址），box 内维字节数不能超过 swizzle 跨度。
  3. 记账字节 = **整个 box**，包括填零的部分。
  4. cluster 里同 x（同一段 M）的 CTA 共享 A，同 y 共享 B；multicast 让每份只从 L2 读一次。

**验收要点**：普通 box 是行主序拷贝 · 越界填零 · 负坐标 · swizzle 偏移已知值 · 三种 swizzle 写进去都能按同一规则读回 · 128B swizzle 真的移动了数据 · 非法配置报错 · 字节数含越界部分 · tile 覆盖含尾块 · 2×2 / 2×1 / 1×1 的 mask · multicast 前后的 L2 流量。

## 直觉与类比

::: callout analogy 仓库的叉车和团购
没有 TMA 时，搬一个 tile 像 128 个店员每人跑一趟仓库拿一小箱（每人 16 B），还得自己核对箱子是不是在货架范围内。TMA 是一台叉车：店长写一张单子（描述符 + 坐标），叉车一次把整托盘搬到店里，超出货架的位置自动放空箱（填零），摆放方式按约定的“错位摆法”（swizzle），搬完在门口的计数器上记一笔“到货 N 字节”（complete_tx）。multicast 是**团购**：同一条街（cluster）上几家店要同一批货，叉车只去仓库（L2）一趟，在每家门口各卸一份。
:::

## 图解：一次 TMA load 做了什么

::: flow 描述符 + 坐标 → SMEM + mbarrier
cfg cw=210 rh=100 nw=172
n desc 0 0 "CUtensorMap" "形状 / box / swizzle" k=dark
n issue 1 0 "1 个线程发指令" "坐标 (row0, col0)" k=acc
n oob 2 0 "越界填零" "kernel 不用判断" k=warn
n swz 3 0 "swizzle 写 SMEM" "16 B 块 ⊕ 行号" k=teal
n bar 3 1 "mbarrier" "complete_tx(box 字节)" k=pass
n mc 2 1 "multicast" "ctaMask 里每个 CTA" k=vio
e desc -> issue
e issue -> oob
e oob -> swz
e swz -> bar
e oob:b -> mc:t
e mc -> bar
cap: 越界、swizzle、记账、multicast 全在硬件里。代价是描述符要在 host 端提前建好（通常放进 kernel 参数或 `__grid_constant__`），而且 box 形状在编译期就要和 SMEM layout 对上。
:::

## 逐步演示

::: steps 8×8 的输出 tile 网格，A、B tile 各 128×64 bf16（16 KB），每步 k 从 L2 读多少
::: flow
cfg cw=200 rh=90 nw=166
n c11 0 0 "无 cluster" "每 CTA 读 A + B" k=todo
n c21 1 0 "cluster 2×1" "B 两家共享" k=warn
n c22 2 0 "cluster 2×2" "A、B 都共享" k=pass
n sm120 3 0 "RTX 5090" "无 multicast" k=dark
e c11 -> c21
e c21 -> c22
:::
--- step 无 multicast {hl=c11}
64 个 CTA × (16 KB + 16 KB) = 2,097,152 B（`l2_bytes_per_kstep(8, 8, a, b)`）。
--- step cluster 2×1 {hl=c11,c21}
cluster 沿 M 方向 2 个 CTA，算同一段 N → 共用 B：64 × (16 KB + 8 KB) = 1,572,864 B，省 25%。mask：`multicast_masks(2, 1)` = `{0: (0b01, 0b11), 1: (0b10, 0b11)}`。
--- step cluster 2×2 {hl=c21,c22}
A、B 都被 2 个 CTA 共享：1,048,576 B，减半。CTA 0 的 A mask = `0b0101`（它和 CTA 2 同 x），B mask = `0b0011`（它和 CTA 1 同 y）。
--- step 5090 {hl=sm120}
sm120 的 TMA 只能写 `shared::cta`，cluster 形状只能 1×1×1（CUTLASS 例子 79a 的 `ClusterShape`）：省掉了线程，但省不了 L2 带宽。
:::

## 易错点

- **越界时读了 `gmem[-1]`。** numpy 的负下标会绕回去！负坐标必须显式判断（`test_negative_coordinates_are_out_of_bounds`）。
- **按“有效元素数”算 expect_tx。** 尾 tile 的 barrier 永远凑不齐字节（或提前完成）；TMA 总是按 box 记账。
- **swizzle 用元素偏移而不是字节偏移。** TMA 的 swizzle 定义在字节上：bf16 的 128B swizzle 是 `Swizzle<3,4,3>`（字节）= `Swizzle<3,3,3>`（元素）。
- **box 内维超过 swizzle 跨度不报错。** 真实 API 会拒绝；128B swizzle + bf16 的 box 内维最多 64 个元素。
- **mask 只包含别人不包含自己。** multicast 的目标 CTA 集合包含发起者自己。

## 三级提示

::: hint 1 I/O 示例与关键问题
```python
g = np.arange(1, 701).reshape(10, 70)
tma_load_2d(g, 8, 66, 4, 8).reshape(4, 8)[:2, :4]   # == g[8:10, 66:70]，其余 0
swizzle_byte_offset(128, 128)       # 144：第 1 行的块 0 去了块 1
tx_bytes(64, 64)                    # 8192
multicast_masks(2, 2)[0]            # (0b0101, 0b0011)
```

#### 关键问题
1. box 的第 r 行第 c 列，在 SMEM 里的字节偏移是多少（不 swizzle / 128B swizzle）？
2. 为什么 swizzle 的“行号”取的是字节偏移的第 7–9 位？
3. 2×2 cluster 里，CTA 3 的 A mask 和 B mask 是多少？
4. multicast 省的是哪一段带宽？对 HBM 有帮助吗？
:::

::: hint 2 思路与常见坑
- `swizzle_byte_offset`：`B = {32:1, 64:2, 128:3}[swz]`，`mask = ((1<<B)-1) << 7`，`o ^ ((o & mask) >> 3)`。
- `tma_load_2d`：先校验；两重循环，越界判断用 `0 <= gr < H and 0 <= gc < W`；`smem[swizzle(o) // elem_bytes] = v`。
- `multicast_masks`：rank = `x + y·cm`；A mask 固定 x、遍历 y；B mask 固定 y、遍历 x。
- `l2_bytes_per_kstep`：先查整除，再 `m·n·(a/cn + b/cm)`。
:::

::: hint 3 详解：答案
::: answer 关键问题答案
1. `o = (r·box_cols + c)·elem_bytes`；128B swizzle 下是 `swizzle_byte_offset(o, 128)`：16 B 块号 `(o >> 4) & 7` 异或 `(o >> 7) & 7`。
2. 128B swizzle 的“行”就是 128 字节：第 7 位以上是 128 B 行号，取低 3 位（8 行一个循环，即 1024 B 的 swizzle atom）。
3. CTA 3 = (x=1, y=1)：A mask = CTA 1 和 3 = `0b1010`；B mask = CTA 2 和 3 = `0b1100`。
4. 省的是 L2 → SM 的带宽（和 L2 的读端口压力）。HBM 流量取决于 L2 命中率；同一时刻多个 CTA 读同一块时本来就大多命中 L2，multicast 主要是让 L2 不成为瓶颈。
:::
:::

## 参考解

::: solution solutions/b3_tma_box_copy.py
{{code: solutions/b3_tma_box_copy.py}}
:::

## 面试官追问 & 对照源码

- **“TMA store 呢？”** 反方向的 `cp.async.bulk.tensor … .global.shared::cta`，Hopper epilogue 用它把 C 从 SMEM 写回，和下一个 tile 的 mainloop 重叠（`tma_store_wait` 控制 SMEM 何时可复用）。
- **“TMA 能做 im2col 吗？”** 能：`cp.async.bulk.tensor.*.im2col` 模式，卷积 kernel 用它省掉显式 im2col。
- **“Rubin 的 TMA 有什么新东西？”** CUTLASS 4.9 的例子 121 用“数据有效性报告”代替 PDL 做 kernel 间同步：TMA 按 16 B 报告上游写的数据是否已经有效，见 [[cutlass:examples/121_rubin_data_driven_sync_gemm/121_rubin_data_driven_sync_gemm.cu|121_rubin_data_driven_sync_gemm.cu]]（C4 讲 PDL）。
- **“DeepGEMM 为什么有时关掉 multicast？”** multicast 要求 cluster 内 CTA 同时推进，会让负载不均的尾 tile 更难调度；小矩阵时收益也小。
