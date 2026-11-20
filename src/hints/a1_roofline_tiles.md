lede: 每一代 GPU 都在回答同一个问题：Tensor Core 又快了一倍，数据从哪来、tile 要多大才喂得饱它。先把这笔账算清楚，后面所有题目都在解决这笔账里的某一项。
# A1 · Roofline、tile 算术强度与寄存器 / SMEM 预算

## 题面速览

- **在系统中的位置**：CUTLASS 选 tile 形状（`TileShape`、`ClusterShape`）、FlashAttention 选块大小、调 occupancy 时，心里算的都是这几笔账。它们不在某个函数里，而是在每个 kernel 的模板参数里，例如 CUTLASS 例子 [[cutlass:examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu@ThreadBlockShape|79a 的 ThreadBlockShape]]。
- **你要实现**：`machine_balance`、`smem_balance`、`tile_intensity`、`gemm_intensity`、`attainable_tflops`、`smallest_feeding_tile`、`accumulator_regs_per_thread`、`ctas_per_sm`。规格表 `GPU` / `GPUS` 已经给出。
- **难点在这几条规则**：
  1. 有**两本账**：整卡的“FLOP / HBM 字节”和每个 SM 的“FLOP / SMEM 字节”，后者决定 tile 最小多大。
  2. tile 每走一步 k 的强度是 **面积 / 周长**：`M·N / (M + N)`（BF16）。
  3. 寄存器按 warp 分配，**每线程向上取整到 8**；超过 255 个就根本放不下。
  4. 一个 SM 能放几个 CTA，是**四个上限取最小**：寄存器、SMEM、线程、CTA 个数。

**验收要点**：H100 机器平衡约 295 · SMEM 平衡 8/16/32/64、5090 是 4 · 64×64 → 32、128×256 → 85.3 · decode GEMV 强度约 1 · roofline 取最小 · 每代最小喂饱 tile · 累加器寄存器向上取整 · 33 个寄存器按 40 算 · 5090 每 CTA 最多 99 KB。

## 直觉与类比

::: callout analogy 厨房里的灶台和备菜台
Tensor Core 是一排猛火灶，SMEM 是灶台旁边的备菜台，HBM 是楼下的仓库。每一代灶台的火力翻倍（Tensor FLOP/clk），但备菜台和楼梯的宽度（SMEM 128 B/clk、HBM 带宽）几乎不变。想让灶不停火，只有一个办法：**每拿一份菜，炒更多的盘子**。一个 M×N 的 tile 就是一锅：拿一列 A（M 份）和一行 B（N 份），能炒 M·N 盘。锅越大越方，每份菜炒的盘子越多——这就是 `M·N / (M + N)`。代价是锅里装的东西（累加器）越来越多，最后连“手里”（寄存器）都端不住，Blackwell 只好专门造一个放锅的架子（TMEM）。
:::

## 图解：两本账

::: flow 一个 GEMM 的数据从哪来、在哪被重复用
cfg cw=210 rh=100 nw=176
n hbm 0 0 "HBM" "H100 3.35 TB/s" k=dark
n l2 1 0 "L2" "跨 CTA 复用" k=vio
n smem 2 0 "SMEM tile" "每 SM 128 B/clk" k=teal
n tc 3 0 "Tensor Core" "H100 4096 FLOP/clk/SM" k=acc
n acc 3 1 "累加器" "RF → TMEM" k=warn
n bal1 0.5 1 "整卡账：295" "FLOP / HBM 字节" k=ghost
n bal2 2 1 "每 SM 账：32" "FLOP / SMEM 字节" k=ghost
e hbm -> l2
e l2 -> smem
e smem -> tc
e tc -> acc
cap: L2 → SMEM 靠 `cp.async` / TMA，SMEM → Tensor Core 靠 `ldmatrix` / wgmma 描述符。整卡账决定 kernel 是不是 memory-bound（`gemm_intensity` vs `machine_balance`）；每 SM 账决定 tile 最小多大（`tile_intensity` vs `smem_balance`）。累加器随 tile 面积增长，是第三本账（寄存器 / TMEM）。
:::

| GPU | 峰值 BF16 TF/s | FLOP / HBM 字节 | FLOP / SMEM 字节 | 最小喂饱 tile（候选里） |
|---|---|---|---|---|
| V100 | 125.3 | 139.3 | 8 | 16×16 |
| A100 | 311.9 | 153.0 | 16 | 32×32 |
| H100 | 989.4 | 295.4 | 32 | 64×64 |
| B200 | 2255.1 | 281.9 | 64 | 128×128 |
| RTX 5090 | 209.8 | 117.1 | 4 | 8×8 |

（数字由参考解算出；峰值按公版 boost 时钟反推，实测时钟往往更高。）RTX 5090 这一行很重要：GeForce 的 FP32 累加 Tensor Core 吞吐只有 H100 的约 1/5，所以**在 5090 上 Ampere 风格的 `mma.sync` 依然够用**，这也是 G 题能在 5090 上跑到 cuBLAS 九成的原因。

## 逐步演示

::: steps 从 V100 到 B200：tile 怎么一步步变大
::: flow
cfg cw=190 rh=112 nw=160
n v 0 0 "V100 · 8" "warp tile 够了" k=pass
n a 1 0 "A100 · 16" "64×64 → 32" k=pass
n h 2 0 "H100 · 32" "64×256 → 51" k=acc
n b 3 0 "B200 · 64" "128×256 → 85" k=vio
n p 3 1 "2-CTA 256×256" "每 SM 128" k=warn
e v -> a
e a -> h
e h -> b
e b -> p "累加器放不下"
:::
--- step V100：8 FLOP/字节 {hl=v}
一个 warp 的 64×64 tile 强度 32，远超需要。瓶颈不在 SMEM，而在“搬运占线程”。
--- step A100：16 {hl=a}
64×64 的 warp tile 仍然够（32 ≥ 16）。问题转向延迟：`cp.async` 多级流水（B1、G2）。
--- step H100：32 {hl=a,h}
64×64 刚好 32，再加上 epilogue、地址计算就不够了。`wgmma` 让一个 warpgroup 共享 64×256 的 B（51.2）。累加器 64×256 FP32 = 每线程 128 个寄存器（`accumulator_regs_per_thread(64, 256, 128)`）。
--- step B200：64 {hl=h,b}
要 128×128 以上。128×256 的累加器每线程 256 个寄存器 > 255：**放不下**。于是有了 TMEM（C1）。
--- step 2-CTA：每 SM 128 {hl=b,p}
两个 SM 合算 256×256，每个 SM 只装一半 B：每 SM 强度 128，是 B200 需要的两倍，留出余量给 epilogue 和尾 tile。
:::

## 易错点

- **用 `M·N / (M + N)` 却忘了字节数。** FP8 的操作数只有 1 字节，同样的 tile 强度翻倍；通用公式是 `2·M·N / ((M+N)·elem_bytes)`。测试 `test_tile_intensity_matches_area_over_perimeter` 会检查 FP8。
- **寄存器不按 8 取整。** 33 个寄存器实际按 40 分配，A100 上 256 线程的 CTA 从 8 个掉到 6 个。`test_ctas_per_sm_register_granularity` 抓这个。
- **忘了 SMEM 每 CTA 上限。** 5090 每 SM 100 KB，但每 CTA 最多 99 KB；超过直接是 0 个 CTA（launch 失败），而不是“少放几个”。
- **把 `smem_per_cta == 0` 当成除零。** 不用 SMEM 的 kernel 不受 SMEM 限制。
- **面积相同不按 M 排序。** 题面规定 tie 时 M 小的优先，测试会检查 `(64,128)` vs `(128,64)`。

## 三级提示

::: hint 1 I/O 示例与关键问题
```python
H = GPUS["H100"]
machine_balance(H)            # 295.4
smem_balance(H)               # 32.0
tile_intensity(128, 256)      # 85.33
gemm_intensity(1, 4096, 4096) # 1.0   decode 的 GEMV
attainable_tflops(H, 1.0)     # 3.35
ctas_per_sm(GPUS["RTX5090"], 256, 128, 40960)   # 2
ctas_per_sm(GPUS["RTX5090"], 256, 128, 61440)   # 1
```

#### 关键问题
1. 一步 k 读多少字节、做多少 FLOP？
2. 为什么 decode（M = 1）的 GEMM 不管怎么写都是 memory-bound？
3. 寄存器为什么是“每 warp 256 个”的粒度，而不是每线程 1 个？
4. G2 里 3 级流水（60 KB SMEM）为什么在 5090 上比 2 级（40 KB）慢？
:::

::: hint 2 思路与常见坑
- 先写 `peak = num_sms · clock · tc_flops_per_clk_per_sm`（`GPU.peak_tflops` 已给出）。
- `tile_intensity`：分子 `2·M·N`，分母 `(M+N)·elem_bytes`。
- `smallest_feeding_tile`：`sorted(candidates, key=lambda t: (t[0]*t[1], t[0]))`，找第一个 ≥ `smem_balance` 的。
- `ctas_per_sm`：先处理两个“直接 0”的情况（>255 寄存器、超过每 CTA SMEM 上限），再算四个上限取 `min`。
- 每 warp 寄存器 = `ceil(regs/8)·8·32`；CTA 的 warp 数 = `ceil(threads/32)`。
:::

::: hint 3 详解：答案
::: answer 关键问题答案
1. 读 A 的一列 M 个、B 的一行 N 个元素，共 `(M+N)·elem_bytes` 字节；做 M·N 次乘加 = `2·M·N` FLOP。
2. M = 1 时每个权重元素只被用一次：强度约 1 FLOP/字节，H100 上最多 3.35 TF/s，是峰值的 0.3%。所以 decode 要么批大（M 变大），要么靠 FP8/FP4 权重减字节，要么靠 kernel 之间少停顿（C4、G4）。
3. 寄存器堆按 warp 分配（每个 warp 是一个整体调度单位），硬件分配粒度是 256 个寄存器 = 每线程 8 个。
4. 5090 每 SM 只有 100 KB SMEM：60 KB 的 CTA 只能放 1 个，40 KB 能放 2 个（`ctas_per_sm` 给出 1 和 2）。5090 的 Tensor Core 不快，2 个 CTA 交替就能藏住延迟；3–4 级流水换来的是 occupancy 减半。G2 的 bench 正好能看到这个：2 级约 0.9× cuBLAS，3/4 级约 0.8×。
:::
:::

## 参考解

::: solution solutions/a1_roofline_tiles.py
{{code: solutions/a1_roofline_tiles.py}}
:::

## 面试官追问 & 对照源码

- **“H100 上一个 bf16 GEMM 的 M = N = K = 4096，理论上是不是 compute-bound？”** `gemm_intensity` ≈ 1365 ≫ 295，是。但如果 tile 太小、L2 命中差，实际从 HBM 读的远多于理想值——这就是 C3 的 raster swizzle 要解决的。
- **“FP8 GEMM 的 tile 为什么可以小一些？”** 操作数字节减半，同样 tile 强度翻倍；但 Tensor Core 吞吐也翻倍，所以需要的 tile 其实差不多。
- **“Blackwell 2-CTA MMA 是怎么省 SMEM 带宽的？”** 见 C1 的 `per_sm_intensity` 和 [[cutlass:include/cute/arch/mma_sm100_umma.hpp@SM100_MMA_F16BF16_2x1SM_SS|SM100_MMA_F16BF16_2x1SM_SS]]（`cta_group::2`）。
- **“Hopper 怎么在 warpgroup 之间挪寄存器？”** `setmaxnreg`，见 B4 和 [[cutlass:include/cutlass/arch/reg_reconfig.h@warpgroup_reg_alloc|warpgroup_reg_alloc]]。
- **“5090 上的 TMEM / wgmma 呢？”** sm120 没有：MMA 仍是 warp 级 `mma.sync`（加了 FP4/FP6/FP8 block scale），见 [[cutlass:include/cute/arch/mma_sm120.hpp@SM120_16x8x32_TN|SM120_16x8x32_TN]]。
