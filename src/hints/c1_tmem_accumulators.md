lede: B200 每 SM 每时钟 8192 FLOP，128×256 的 FP32 累加器已经占半个寄存器堆。Blackwell 的回答是 Tensor Memory：每 SM 256 KB 专门放累加器，像 malloc 一样按列申请；512 列正好放下两个累加器，于是 MMA 和 epilogue 天然重叠。
# C1 · Blackwell TMEM：列分配、双缓冲累加器与 2-CTA MMA

## 题面速览

- **在系统中的位置**：TMEM 分配器 [[cutlass:include/cute/arch/tmem_allocator_sm100.hpp@Allocator1Sm|Allocator1Sm]]（`tcgen05.alloc`，2 的幂、32..512 列，一个 warp 发起）；累加器流水 [[cutlass:include/cutlass/pipeline/sm100_pipeline.hpp@PipelineUmmaAsync|PipelineUmmaAsync]]，在 kernel 里叫 `AccumulatorPipeline`（[[cutlass:include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp@AccumulatorPipelineStageCount|AccumulatorPipelineStageCount]]）；2-CTA MMA 是 [[cutlass:include/cute/arch/mma_sm100_umma.hpp@SM100_MMA_F16BF16_2x1SM_SS|SM100_MMA_F16BF16_2x1SM_SS]]（`tcgen05.mma.cta_group::2`）。**真跑需要 B200 / GB200（sm100a）**；RTX 5090（sm120）没有 TMEM 和 tcgen05。
- **你要实现**：`TmemAllocator`（alloc / free / used）、`acc_columns`、`max_acc_buffers`、`acc_pipeline`、`per_sm_intensity`。
- **难点在这几条规则**：
  1. TMEM = 128 lane × 512 列 × 32 bit；累加器的 M 对应 lane、N 对应列：128×256 FP32 = 256 列。
  2. 分配是 2 的幂、≥ 32 列；本题用“按大小对齐的 first-fit”作为放置模型，失败返回 None 且不改状态。
  3. MMA warp 写累加器 i % buffers 之前，要等 tile i − buffers 的 epilogue 读完它。
  4. 2-CTA：两个 SM 合算 256×N，每个 SM 只装一半 B，每 SM 强度从 85 升到 128。

**验收要点**：两个 256 列塞满 · 对齐的 first-fit · 释放后复用 · 非法大小 · 非法释放 · 失败不改状态 · Rubin 576 列 · 列数与缓冲数 · 单缓冲串行 · 双缓冲重叠 · 缓冲越多不越慢 · 2-CTA 强度。

## 直觉与类比

::: callout analogy 两个托盘的流水线
厨师（MMA warp，只有一个线程发指令）把菜炒进托盘（TMEM 里的累加器），服务员（epilogue warp）把托盘上的菜装盘送走（TMEM → 寄存器 → SMEM → TMA store）。只有一个托盘时，厨师炒完得等服务员送完才能炒下一锅。TMEM 有 512 列，一个 128×256 的托盘占 256 列——**正好两个托盘**：厨师炒第二锅的时候服务员在送第一锅。这就是 Hopper 要靠 pingpong（把锅切小、两个厨师轮流）才能做到的事，Blackwell 用“多一个托盘”就做到了，而且锅不用变小。
:::

## 图解：Blackwell GEMM 的数据路径

::: flow tcgen05：操作数在 SMEM，累加器在 TMEM
cfg cw=210 rh=100 nw=172
n tma 0 0 "TMA load warp" "A、B → SMEM" k=acc
n mma 1 0 "MMA warp" "1 个线程发 tcgen05.mma" k=warn
n t0 2 0 "TMEM 累加器 0" "列 0–255" k=vio
n t1 2 1 "TMEM 累加器 1" "列 256–511" k=vio
n epi 3 0.5 "epilogue warps" "tcgen05.ld → TMA store" k=pass
e tma -> mma
e mma -> t0
e mma -> t1
e t0 -> epi
e t1 -> epi
cap: 完成通知：MMA 读完 SMEM 后用 `tcgen05.commit` arrive 一个 mbarrier（释放 SMEM stage），累加器写完再 arrive 一个（通知 epilogue）。epilogue 读完 arrive 另一个（释放累加器）。三个环都是 B2 的 mbarrier 环。
:::

## 逐步演示

::: steps 4 个 tile，ML = 10，EP = 6
::: flow
cfg cw=200 rh=90 nw=166
n one 0 0 "1 个累加器" "每 tile 16 → 64" k=todo
n two 1 0 "2 个累加器" "→ 46" k=pass
n long 2 0 "EP = 15 > ML" "epilogue 成瓶颈 → 70" k=warn
n pair 3 0 "2-CTA 256×256" "每 SM 强度 128" k=vio
e one -> two
e two -> long
:::
--- step 单缓冲 {hl=one}
tile 1 的 MMA 必须等 tile 0 的 epilogue（16）结束：每个 tile 10 + 6，4 个 = `acc_pipeline(4, 10, 6, 1)` = 64。8 个 tile = 128。
--- step 双缓冲 {hl=one,two}
tile 1 写累加器 1，不用等；tile 2 写累加器 0，要等 tile 0 的 epilogue（16），它早在 MMA 跑到 20 之前就结束了：总时间 46。8 个 tile = 86（稳态每 tile 10）。
--- step epilogue 比 MMA 长 {hl=two,long}
EP = 15：epilogue 只有一组 warp、串行处理，总时间 = 10 + 4·15 = 70。这时该加 epilogue warp，或者把 epilogue 切成子 tile（CUTLASS 的 `NumEpilogueSubTiles`）。
--- step 2-CTA {hl=pair}
`per_sm_intensity(128, 256)` = 85.3；`per_sm_intensity(256, 256, cta_pair=True)` = 128。B200 要 64（A1），2-CTA 给了一倍的余量。
:::

## 易错点

- **alloc 失败时已经标记了一部分列。** `test_failed_alloc_changes_nothing`。
- **对齐忘了。** 32..63 空着也不能放 64 列的块（本题的放置模型），`test_alignment_first_fit`。
- **free 不校验。** 释放一个从没分配过的区间、或重复释放，都应报错。
- **双缓冲时 MMA 等错了 tile。** 要等的是 `i − buffers` 的 epilogue，不是 `i − 1`。
- **epilogue 并行处理多个 tile。** 本题只有一组 epilogue warp，必须按顺序做（`max(ml_end[i], ep_end[i-1])`）。

## 三级提示

::: hint 1 I/O 示例与关键问题
```python
a = TmemAllocator()
a.alloc(256), a.alloc(256), a.alloc(32)   # (0, 256, None)
acc_columns(256), max_acc_buffers(256)    # (256, 2)
acc_pipeline(4, 10, 6, 1), acc_pipeline(4, 10, 6, 2)   # (64.0, 46.0)
per_sm_intensity(256, 256, cta_pair=True)  # 128.0
```

#### 关键问题
1. 128×256 的 FP32 累加器为什么占 256 列而不是 128？
2. 双缓冲的稳态周期是多少？
3. 一个 SM 上同时跑两个 CTA，各要两个 256 列的累加器，会发生什么？
4. 2-CTA MMA 为什么能省一半的 B？
:::

::: hint 2 思路与常见坑
- 分配器：一个长度 capacity 的布尔数组 + 已分配集合 `{(start, ncols)}`；alloc 从 0 开始按步长 ncols 扫描。
- `acc_pipeline`：两个列表 `ml_end`、`ep_end`；`s = max(ml_end[-1], ep_end[i-buffers])`；`es = max(ml_end[i], ep_end[-1])`。
- `per_sm_intensity`：pair 时 `m = tile_m / 2`，强度 `m·n / (m + n/2)`。
:::

::: hint 3 详解：答案
::: answer 关键问题答案
1. 128 行对应 128 个 lane（每 lane 一行），每行 256 个 FP32，每列 32 bit：256 列。
2. `max(ML, EP)`：MMA 每 ML 出一个累加器，epilogue 每 EP 消耗一个。
3. 第二个 CTA 分不到（512 列已满），`tcgen05.alloc` 会等；如果第一个 CTA 不及时 `relinquish_alloc_permit` / 释放，就成了串行。所以 CUTLASS 的 persistent kernel 每 SM 只跑一个 CTA，用 tile 调度器循环取 tile（C3）。
4. MMA 读的 B 是两个 SM 的 SMEM 拼起来的：每个 SM 装 N/2 列，硬件通过 cluster 内的 DSMEM 把对方那一半也喂进 Tensor Core。A 各自装自己的 128 行。
:::
:::

## 参考解

::: solution solutions/c1_tmem_accumulators.py
{{code: solutions/c1_tmem_accumulators.py}}
:::

## 面试官追问 & 对照源码

- **“从 Hopper kernel 迁到 Blackwell 要改什么？”** MMA 发起方从 “consumer warpgroup 的 128 个线程执行 wgmma” 变成 “一个 MMA warp 里 elect 一个线程执行 tcgen05.mma”；累加器从寄存器 fragment 变成 TMEM tensor（开头 alloc、结尾 free）；完成通知从 `wgmma.wait_group` 变成 `tcgen05.commit` 到 mbarrier；epilogue 先 `tcgen05.ld` 把 TMEM 读进寄存器。
- **“FA4 把什么放进了 TMEM？”** S、P 和 O 都在 TMEM：QKᵀ 写 S 到 TMEM，softmax warp 读 S、写 P（BF16，覆盖在 S 的位置），PV 的 A 操作数直接从 TMEM 读，O 也在 TMEM，rescale 由 correction warp 做（D3）。见 [[fa:flash_attn/cute/flash_fwd_sm100.py|flash_fwd_sm100.py]]。
- **“DeepGEMM 的 scale factor 也进 TMEM 吗？”** 是：block-scaled MMA 的 scale 用 `tcgen05.cp`（UTCCP）从 SMEM 拷进 TMEM，MMA 指令描述符里带 scale 的 TMEM 地址，见 [[deepgemm:deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh|sm100_fp8_fp4_gemm_1d1d.cuh]]（C2 讲 block scaling）。
- **“Rubin 有什么不同？”** CUTLASS 里 Sm107 的 TMEM 是 576 列，`exclusive` 分配可以是任意 32 的倍数（见 Allocator1Sm 的注释）。
