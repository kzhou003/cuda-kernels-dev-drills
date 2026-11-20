lede: FP8 / FP4 让 Tensor Core 吞吐翻倍再翻倍，代价是 4 bit 只能表示 ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}。能用的前提是“每一小块共享一个 scale”。Blackwell 在 Tensor Core 里直接乘 scale；Hopper 上 DeepGEMM 每 128 个 K 在 CUDA core 上乘，同时绕开了 FP8 累加器精度不够的问题。
# C2 · MXFP8 / NVFP4 block scaling 与 DeepGEMM 的 FP8 提升

## 题面速览

- **在系统中的位置**：Blackwell 的 `tcgen05.mma.kind::mxf8f6f4 / mxf4nvf4 .block_scale`（[[cutlass:include/cute/arch/mma_sm100_umma.hpp@SM100_MMA_MXF4_SS|SM100_MMA_MXF4_SS]]，scale 在 TMEM 里）；RTX 5090 的 warp 级版本 [[cutlass:include/cute/arch/mma_sm120.hpp@SM120_16x8x32_TN|SM120_16x8x32_TN]]（`mma.sync … block_scale`，sm120 上能真跑）；格式类型 [[cutlass:include/cutlass/float_subbyte.h@float_e2m1_t|float_e2m1_t]]、[[cutlass:include/cutlass/float8.h@float_ue8m0_t|float_ue8m0_t]]。Hopper 上的软件做法：[[deepgemm:deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh|DeepGEMM sm90_fp8_gemm_1d2d.cuh]]。
- **你要实现**：`quantize_e2m1`、`mx_quantize`（MXFP8 / MXFP4）、`nvfp4_quantize`、`dequantize`、`block_scaled_dot`、`promoted_dot`。`round_e4m3`、`round_mantissa` 已给出。
- **难点在这几条规则**：
  1. E2M1 的格点**不均匀**（间隔 0.5 / 0.5 / 1 / 2），不能用“除以步长再 round”；平局取尾数位为 0 的那个。
  2. MX：每 32 个元素一个 2 的幂 scale = `2^(floor(log2 amax) − emax)`，emax 是元素格式的最大指数（E4M3 是 8，E2M1 是 2）。
  3. NVFP4：每 16 个元素一个 **E4M3** scale（有尾数）+ 一个全张量 FP32 scale，两级。
  4. 低精度累加器加久了会“吞掉”小的加数；每隔 128 个 K 把部分和提升到高精度，误差降几个数量级。

**验收要点**：E2M1 格点与符号 · 平局取偶 · 就近与饱和 · MX E4M3 的 scale 是 2 的幂且等于公式 · 一个已知 E2M1 块（7 饱和成 6）· 零块与报错 · NVFP4 误差小于 MXFP4 · NVFP4 的全局 scale 与零块 · block-scaled 点积 = 反量化后的点积 · 提升修复 swamping · 提升边界。

## 直觉与类比

::: callout analogy 每一页一个比例尺的地图册
FP4 像一张只有 8 个刻度的尺子（0、0.5、1、1.5、2、3、4、6）。想用它量一整个国家，只能每一页地图配一个比例尺：这一页是城市（scale 小），那一页是沙漠（scale 大），尺子还是那把尺子。MX 格式的比例尺只能是 2 的幂（1:1、1:2、1:4……），所以经常“刚好差一点”，最大的那个值被截掉；NVFP4 的比例尺本身就是一个 FP8 数（可以是 1:1.17），而且每 16 个元素就换一页，量得更准。DeepGEMM 的“提升”则像记账：在小本子上（Tensor Core 里精度有限的累加器）记 128 笔就誊到总账（FP32）上，小本子永远不会记到“大数吃掉小数”的地步。
:::

## 图解：一次 block-scaled 点积

::: flow 量化 → Tensor Core → 乘 scale → FP32
cfg cw=210 rh=100 nw=172
n x 0 0 "BF16 激活 / 权重" "按 16 或 32 分块" k=dark
n s 1 0 "每块一个 scale" "UE8M0 或 E4M3" k=warn
n q 2 0 "FP4 / FP8 元素" "E2M1 / E4M3" k=acc
n mma 3 0 "块内乘加" "Tensor Core" k=vio
n acc 3 1 "× sa·sb → FP32" "Blackwell：在 TC 里" k=pass
n dg 2 1 "Hopper：wgmma" "每 128 K 在 CUDA core 乘" k=teal
e x -> s
e s -> q
e q -> mma
e mma -> acc
e q:b -> dg:t
e dg -> acc
cap: Blackwell 的 block-scaled MMA 在 Tensor Core 里做“块内乘加 × 两边的 scale”；Hopper 的 wgmma 不认识 scale，DeepGEMM 让 wgmma 累加一个 128-K 的块，再在 CUDA core 上 `final += sa·sb·partial`（[[deepgemm:deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh#L308-L344|L308–344]]）。
:::

## 逐步演示

::: steps 一块数据 [0.12, −0.5, 3.1, 7.0, 0.9, −2.2, 0, 1.6, …]
::: flow
cfg cw=200 rh=90 nw=166
n amax 0 0 "amax = 7" "块内最大绝对值" k=dark
n mx 1 0 "MXFP4" "scale 2^0 = 1" k=warn
n nv 2 0 "NVFP4" "scale = 7/6" k=pass
n acc 3 0 "累加" "提升 vs 不提升" k=vio
e amax -> mx
e mx -> nv
e nv -> acc
:::
--- step E2M1 的取整 {hl=amax}
`quantize_e2m1([0.25, 0.75, 1.25, 2.5, 5.0, 7.0, −0.3])` = `[0, 1, 1, 2, 4, 6, −0.5]`：0.25 在 0 和 0.5 中间，取 0（尾数位 0）；2.5 取 2；5.0 取 4；7 饱和成 6。
--- step MXFP4 {hl=mx}
`floor(log2 7) − 2 = 0` → scale 1。元素 `[0, −0.5, 3, 6, 1, −2, 0, 1.5]`。7.0 变成 6（误差 1），3.1 变成 3。
--- step NVFP4 {hl=nv}
这一块（16 个元素）的 E4M3 scale × 全局 scale = 7/6：元素同样是 `[0, −0.5, 3, 6, 1, −2, 0, 1.5]`，但还原后是 `[0, −0.58, 3.5, 7.0, 1.17, −2.33, 0, 1.75]`：最大值 7 被精确保留。在 4096 个随机数上：平均绝对误差 NVFP4 0.106、MXFP4 0.126、MXFP8 0.028（|x| 均值 1.19）。
--- step 提升 {hl=acc}
8192 个 [0.5, 1.5] 的乘积求和：13 位尾数的累加器不提升，相对误差 8.8e−4；每 128 个提升一次 6.2e−7。10 位尾数不提升时，和加到一定大小就“加不动了”，误差 47%；每 128 个提升一次 1.2e−4。
:::

## 易错点

- **E2M1 用均匀步长取整。** 例如 `round(x / 0.5) * 0.5` 会产生 2.5、3.5、5 这些不存在的值。
- **平局取了下标较小的那个。** 0.75 应该是 1（尾数位 0），不是 0.5（`test_e2m1_ties_go_to_even_mantissa`）。
- **MX 的 scale 用了 `ceil` 或 `log2(amax / 6)`。** OCP MX 规范是 `floor(log2 amax) − emax`，所以最大值可能饱和——这是格式本身的特性，测试按规范检查。
- **零块的 scale 算出 −inf。** amax = 0 时 scale = 1（MX）或 s = 0、q 全 0（NVFP4）。
- **提升边界从 0 数起。** 第 `promote_every` 个累加之后提升（`(k + 1) % step == 0`），最后再提升一次剩下的。

## 三级提示

::: hint 1 I/O 示例与关键问题
```python
quantize_e2m1([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])   # [0, 1, 1, 2, 2, 4, 4]
q, s = mx_quantize(np.array([7.0, 1.0, -0.4, 0.0] * 8), 32, "e2m1")
s, q[:4]                                                  # [1.], [6, 1, -0.5, 0]
q, s, g = nvfp4_quantize(np.concatenate([np.zeros(16), np.linspace(-3, 3, 16)]))
g, s                                                      # 3/2688, [0, 448]
```

#### 关键问题
1. E2M1 的 8 个非负值里，哪些是“尾数位为 0”的？
2. 为什么 MX 的 scale 选 `floor(log2 amax) − emax` 而不是让 amax 正好对上最大值？
3. NVFP4 为什么要两级 scale？
4. 为什么提升间隔是 128，而不是 1（每次都提升）或 K（不提升）？
:::

::: hint 2 思路与常见坑
- `quantize_e2m1`：`a = min(|x|, 6)`；对 8 个格点算距离 `d`；找最小距离的候选；候选里优先偶数下标；乘回符号。
- `mx_quantize`：`reshape(-1, block)`，逐块 amax、指数、clip 到 [−127, 127]，`q = rnd(xb / scale)`。
- `nvfp4_quantize`：先 g，再 `s = round_e4m3(amax_b / (6g))`，`s == 0` 的块 q 全 0（避免除以 0）。
- `promoted_dot`：`acc = round_mantissa(acc + a[k]*b[k], bits)`；到点就 `total += acc; acc = 0`。
:::

::: hint 3 详解：答案
::: answer 关键问题答案
1. 下标偶数的：0、1、2、4（0.5、1.5、3、6 的尾数位是 1）。
2. scale 必须是 2 的幂（UE8M0 只有指数），它只能把 amax 放进“最大指数那一档”；amax 落在这一档的上半段时会超过 6 而饱和。规范接受这点误差，换来只用 8 bit 存 scale、硬件乘 scale 只是加指数。
3. E4M3 的 scale 自己的动态范围有限（最大 448）。先用全张量 FP32 scale g 把整个张量缩放到 “E4M3 scale × E2M1 元素” 能表示的 448·6 以内，块 scale 再负责块内的细节。
4. 每次都提升等于不用 Tensor Core 的累加（太慢）；完全不提升会 swamping。128 个 K 一次正好对应 DeepGEMM 里 wgmma 一个 K 块（同时也是 FP8 的 scale 粒度 128），提升的开销被摊到 128 次乘加上。
:::
:::

## 参考解

::: solution solutions/c2_block_scaling.py
{{code: solutions/c2_block_scaling.py}}
:::

## 面试官追问 & 对照源码

- **“FP8 训练里 per-tensor、per-channel、per-block（1×128 / 128×128）scale 怎么选？”** 越细误差越小，但 scale 越多、kernel 越复杂。DeepSeek-V3 用激活 1×128、权重 128×128，正好对应 DeepGEMM 的 “1d2d”。
- **“Hopper FP8 wgmma 的累加器到底多少位？”** DeepSeek-V3 报告说约 14 位有效精度，这就是 DeepGEMM 要做 CUDA-core 提升的原因；本题用 `round_mantissa(…, 13)` 模拟。
- **“5090 上能跑 NVFP4 GEMM 吗？”** 能：sm120 的 `mma.sync … kind::mxf4nvf4.block_scale`，CUTLASS 例子 [[cutlass:examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu|79a]] 就是 NVFP4 × NVFP4 → BF16（cluster 只能 1×1×1）。可以作为 G 题的扩展练习。
- **“B200 上 scale 存在哪？”** 用 `tcgen05.cp` 从 SMEM 拷进 TMEM，MMA 指令描述符里带 scale 的 TMEM 地址（C1 的追问）。
