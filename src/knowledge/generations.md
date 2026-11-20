lede: 从 Volta 到 Blackwell，每一代加了哪几样东西、为什么加、kernel 代码因此怎么变。读完这一页，再看 CUTLASS 里 sm80 / sm90 / sm100 三个目录为什么长得完全不一样。
# 每一代 GPU：加了什么、为什么、代码怎么变

## 问题在哪

| | V100 | A100 | H100 | B200 | RTX 5090 |
|---|---|---|---|---|---|
| 架构 | sm70 | sm80 | sm90a | sm100a | sm120 |
| BF16 Tensor 峰值（TF/s） | 125（FP16） | 312 | 989 | 2255 | 210 |
| HBM / 显存带宽（TB/s） | 0.9 | 2.0 | 3.35 | 8.0 | 1.8 |
| 每 SM FLOP/clk | 1024 | 2048 | 4096 | 8192 | 512 |
| 每 SM FLOP / SMEM 字节 | 8 | 16 | 32 | 64 | 4 |
| 每 CTA 最大 SMEM | 96 KB | 163 KB | 227 KB | 227 KB | 99 KB |

（由 A1 的规格表算出；5090 的数字是 FP32 累加的 dense BF16。）最后两行是关键：每 SM 的“饥饿程度”每代翻倍，而 5090 反而比 V100 还“不饿”——这就是为什么 Ampere 风格的 kernel 在 5090 上依然能跑到 cuBLAS 的九成（G2）。

## Volta / Turing：一条指令一个小矩阵乘

Tensor Core 诞生：`mma.sync` 让一个 warp 合作完成 8×8×4（Volta）、16×8×8（Turing）的矩阵乘。从此写 kernel 变成**数据布局问题**：每个线程寄存器里必须恰好是指令要的元素（A2）。Turing 加了 `ldmatrix`，一条指令把 SMEM 里的 8×8 块读成 fragment 布局，并且“报地址的 lane”和“拿数据的 lane”不是同一套映射。

## Ampere：搬运变成异步的

A100 每 SM 每时钟 2048 FLOP，SMEM 带宽没变。`cp.async` 让线程发出 global → shared 拷贝后不等结果，配合 commit / wait_group 组成 3–5 级 ring buffer（B1、G2）。`ldmatrix + mma.sync m16n8k16` 是这一代的标准组合。剩下的问题：每个线程还要算地址、发拷贝，累加器占满寄存器。

## Hopper：硬件搬运、异步 MMA、CTA 内分工

::: flow Hopper GEMM 的一个 CTA
cfg cw=200 rh=100 nw=166
n tma 0 0 "TMA" "1 个线程 1 条指令" k=acc
n bar 1 0 "mbarrier" "按字节完成" k=warn
n wg 2 0 "wgmma" "warpgroup，读 SMEM" k=vio
n ws 3 0 "warp specialization" "producer / consumer" k=pass
n cl 0 1 "cluster + multicast" "一次 L2 读多家用" k=teal
n reg 3 1 "setmaxnreg" "40 / 232 / 232" k=ghost
e tma -> bar
e bar -> wg
e wg -> ws
e tma:b -> cl:t
e ws:b -> reg:t
cap: B2、B3、B4 分别复刻 mbarrier、TMA、warp specialization。wgmma 的 A/B 从 SMEM 描述符读，所以 SMEM 的 swizzle（A3）必须和 TMA 写入的一致。
:::

- **TMA**：描述符 + 坐标，硬件做越界填零、swizzle、按字节记账；cluster 内 multicast（B3）。
- **mbarrier**：到达计数 + 事务字节 + 相位位，producer / consumer 环（B2）。
- **wgmma**：4 个 warp（一个 warpgroup）一起发，64×N×16，异步，用 `wgmma.commit_group / wait_group` 等。
- **warp specialization**：producer warpgroup 只发 TMA，consumer 只做 MMA 和 epilogue；pingpong 让两个 consumer 轮流用 Tensor Core（B4）。
- **setmaxnreg**：运行时把寄存器从 producer 挪给 consumer。

## Blackwell：累加器离开寄存器堆

::: callout analogy 从“手里端着锅”到“锅放在架子上”
Hopper 的厨师（consumer warpgroup）要一直用手（寄存器）端着锅（累加器），所以锅不能太大，两个厨师还得轮流用灶。Blackwell 给每个灶配了一个专门的架子（TMEM，256 KB），锅放在架子上，一个人按按钮（单线程 `tcgen05.mma`）就能炒，另一个人从另一个架子上装盘（epilogue `tcgen05.ld`）。架子上正好放两口大锅（512 列 = 2 × 256），炒和装盘自然错开（C1）。
:::

- **TMEM** + `tcgen05.mma`（单线程发起）+ `tcgen05.commit`（MMA 完成时 arrive mbarrier）。
- **2-CTA MMA**（`cta_group::2`）：两个 SM 合算 256×N，每个 SM 只装一半 B，每 SM 强度 128（C1）。
- **block-scaled MMA**：MXFP8 / MXFP4 / NVFP4 的 scale 在 Tensor Core 里乘（C2）。
- **Cluster Launch Control**：动态 tile 调度（C3）。

## 同名不同芯：RTX 5090（sm120）

GeForce Blackwell 不是小号 B200：没有 tcgen05、TMEM、TMA multicast；MMA 仍是 warp 级 `mma.sync`（加了 FP4/FP6/FP8 和 block scale：[[cutlass:include/cute/arch/mma_sm120.hpp@SM120_16x8x32_TN|SM120_16x8x32_TN]]）；每 CTA SMEM 99 KB；TMA 目标只能是 `shared::cta`（[[cutlass:include/cute/arch/copy_sm90_tma.hpp#L115-L118|copy_sm90_tma.hpp L115–118]]）。FA3、FA4 和 CUTLASS sm90 / sm100 的核心路径在 5090 上都编译不了。好消息：它支持 PDL、mbarrier、cp.async、TMA（无 multicast）——G 题用到的都有。

## Rubin（CUTLASS 里的 Sm107）：往 kernel 之间走

更大的 SMEM（每 CTA 327 KB）和 TMEM（576 列），以及 TMA 的数据有效性报告：下游 kernel 按 16 字节粒度知道上游的数据写好了没有，可以按 tile 同步而不是等整个 grid（[[cutlass:examples/121_rubin_data_driven_sync_gemm/121_rubin_data_driven_sync_gemm.cu|例子 121]]）。这是 C4 / G4 的 PDL 的下一步。

## 和练习题的对应关系

| 题目 | 哪一代的哪个特性 | 能在 5090 上真跑吗 |
|---|---|---|
| A2、G1 | Turing / Ampere：mma.sync + ldmatrix | 能 |
| B1、G2 | Ampere：cp.async 多级流水 | 能 |
| A3 | 所有代：SMEM swizzle | 能（G 题用 padding） |
| B2 | Hopper+：mbarrier | 能（sm120 有 mbarrier） |
| B3 | Hopper：TMA + multicast | TMA 能，multicast 不能（需 H100 / B200） |
| B4 | Hopper：wgmma、setmaxnreg | 不能（需 H100） |
| C1 | Blackwell：TMEM、tcgen05、2-CTA | 不能（需 B200） |
| C2 | Blackwell block scale / Hopper FP8 | NVFP4 的 mma.sync 能（sm120） |
| C3 | 所有代：tile 调度；CLC 是 Blackwell | 调度能；CLC 需 B200 |
| C4、G4 | Hopper+：PDL；CUDA Graph | 能 |
