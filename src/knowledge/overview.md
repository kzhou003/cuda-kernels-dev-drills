lede: 这套练习在练什么：每一代 NVIDIA GPU 都让 Tensor Core 快一倍，而数据供给跟不上。几乎所有新特性都在回答同一个问题——怎么让更快的矩阵单元不挨饿。GEMM 和 FlashAttention 是两条贯穿始终的主线。
# 总览：一个问题，五个部件，四代答案

## 问题在哪

| 事实 | 后果 | 对应练习 |
|---|---|---|
| Tensor Core 每代 ×2，SMEM 带宽一直是 128 B/clk/SM | tile 必须越来越大：每 SM 的 FLOP/字节从 8 → 16 → 32 → 64 | A1 |
| tile 越大，累加器越多 | 128×256 FP32 = 半个寄存器堆 → setmaxnreg（Hopper）、TMEM（Blackwell） | B4、C1 |
| 每个线程自己搬数据要占线程和寄存器 | cp.async（Ampere）→ TMA（Hopper） | B1、B3、G2 |
| 搬运和计算都异步了，需要“信号灯” | mbarrier：到达计数 + 事务字节 + 相位位 | B2 |
| MMA 也异步了，warp 可以分工 | warp specialization、pingpong | B4 |
| 低精度更快，但 4 bit 只能表示 16 个值 | block scaling：MX、NVFP4；Hopper 上 DeepGEMM 的提升 | C2 |
| kernel 内部越快，kernel 之间的空隙占比越大 | persistent / Stream-K、CUDA Graph、PDL | C3、C4、G4 |
| attention 的 softmax 用的是另一个更慢的单元 | FA3 重叠、FA4 的 FMA 多项式和条件 rescale | D1–D3、G3 |

## 解决办法：一个 tiled GEMM 的五个部件

每一个 Tensor Core GEMM（FlashAttention 也是两个背靠背的 GEMM）都由五个部件组成，每一代换掉其中几个：

::: flow 五个部件
cfg cw=180 rh=100 nw=150
n copy 0 0 "① 搬运" "global → SMEM" k=acc
n sync 1 0 "② 同步" "数据到齐了吗" k=warn
n mma 2 0 "③ MMA" "Tensor Core" k=vio
n pipe 3 0 "④ 流水" "ring buffer" k=teal
n epi 4 0 "⑤ epilogue" "累加器 → HBM" k=pass
e copy -> sync
e sync -> mma
e mma -> pipe
e pipe -> epi
e pipe:b -> copy:b
cap: 部件 ④ 把 ①②③ 串成一个环：搬运永远比计算提前几块。
:::

| 部件 | sm70/75（V100、T4） | sm80（A100、RTX 30/40） | sm90a（H100） | sm100a（B200） | sm120（RTX 5090） |
|---|---|---|---|---|---|
| ① 搬运 | ld.global → st.shared | cp.async（B1、G2） | TMA，可 multicast（B3） | TMA；2-CTA 各搬一半 | cp.async、TMA（无 multicast） |
| ② 同步 | `__syncthreads` | `cp.async.wait_group` | mbarrier + 事务字节（B2） | mbarrier + `tcgen05.commit` | 同 sm80 / sm90 |
| ③ MMA | wmma / mma.sync | mma.sync + ldmatrix（A2、G1） | wgmma，读 SMEM 描述符 | tcgen05.mma，D 在 TMEM（C1） | mma.sync（+ FP4/FP8 block scale） |
| ④ 流水 | 2 级 | 3–5 级 | 4–8 级 + warp specialization（B4） | 同左 + 累加器双缓冲（C1） | 2–4 级（G2） |
| ⑤ epilogue | RF → global | RF → SMEM → global | RF → SMEM → TMA store | TMEM → RF → SMEM → TMA store | RF → global（G1） |

::: callout analogy 一家越开越快的餐厅
灶台（Tensor Core）每年换一次，火力翻倍；但备菜台（SMEM）、楼梯（HBM）、服务员（线程）都没怎么变。每一代的改造都是在“让灶台别停火”：锅换大（tile 变大）、雇专门的搬运工（TMA）、装上叫号灯（mbarrier）、厨师分工（warp specialization）、加一个放锅的架子（TMEM）、提前把下一桌的餐具摆好（PDL）。练习的顺序就是这家餐厅改造的顺序。
:::

## 怎么做这套练习

- **CPU 题（A–D，14 道）**：纯 Python + numpy，每道复刻一个机制的“规则”：布局、协议、调度、数值。不需要 GPU，几秒内跑完。
- **GPU 题（G，4 道）**：真实 CUDA，默认在 **RTX 5090（sm_120）** 上测。G1–G3 在任何 sm_80+ 上都能跑，G4 需要 sm_90+。
- **5090 跑不了的**：wgmma、TMA multicast、`setmaxnreg`、tcgen05 / TMEM、2-CTA MMA 需要 H100（sm_90a）或 B200（sm_100a）。对应的 B3、B4、C1、D3 是 CPU 模型，提示页里写明了需要哪张卡才能真跑、以及在 CUTLASS 里对应的源码。

建议顺序：A1 → A2 → G1 → A3 → B1 → G2 → B2 → B3 → B4 → C1 → C2 → C3 → D1 → D2 → G3 → D3 → C4 → G4。每做完一道 CPU 题，紧接着做对应的 GPU 题，“规则”马上变成“代码”。

## 和练习题的对应关系

| 题目 | 复刻了什么 | 源码 |
|---|---|---|
| A1 | tile 形状和 occupancy 的算账 | [[cutlass:examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu@ThreadBlockShape|79a 的 tile 配置]] |
| A2 | mma.sync / ldmatrix 的 fragment 布局 | [[cutlass:include/cute/arch/mma_sm80.hpp@SM80_16x8x16_F32BF16BF16F32_TN|SM80_16x8x16]] |
| A3 | SMEM swizzle | [[cutlass:include/cute/swizzle.hpp@Swizzle|cute::Swizzle]] |
| B1 | cp.async 多级流水 | [[cutlass:include/cutlass/gemm/collective/sm80_mma_multistage.hpp|sm80_mma_multistage]] |
| B2 | mbarrier 与 TMA pipeline | [[cutlass:include/cutlass/pipeline/sm90_pipeline.hpp@PipelineTmaAsync|PipelineTmaAsync]] |
| B3 | TMA box、swizzle、multicast | [[cutlass:include/cute/arch/copy_sm90_tma.hpp@SM90_TMA_LOAD_MULTICAST_2D|SM90_TMA_LOAD_MULTICAST_2D]] |
| B4 | pingpong / cooperative、setmaxnreg | [[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp|sm90 pingpong kernel]] |
| C1 | TMEM 分配与累加器流水 | [[cutlass:include/cute/arch/tmem_allocator_sm100.hpp@Allocator1Sm|Allocator1Sm]] |
| C2 | MX / NVFP4、FP8 提升 | [[deepgemm:deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh|DeepGEMM sm90 FP8]] |
| C3 | tile 调度与 Stream-K | [[cutlass:include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp|sm90 Stream-K]] |
| C4 | stream / graph / PDL 时间线 | [[cutlass:include/cutlass/arch/grid_dependency_control.h|grid_dependency_control.h]] |
| D1 | online softmax 与 merge | [[fa:csrc/flash_attn/src/softmax.h@softmax_rescale_o|softmax_rescale_o]] |
| D2 | FA2 分块与 causal | [[fa:csrc/flash_attn/src/flash_fwd_kernel.h@compute_attn_1rowblock|compute_attn_1rowblock]] |
| D3 | FA4 的 exp2 模拟与条件 rescale | [[fa:flash_attn/cute/utils.py@ex2_emulation|ex2_emulation]] |
| G1–G4 | 5090 上的真实 kernel | 本仓库 `solutions/*.cu` |
