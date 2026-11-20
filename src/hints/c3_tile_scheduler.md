lede: kernel 内部越快，“哪个 SM 算哪个 tile”越重要：最后一波只有一半 SM 有活干、同时在跑的 tile 共用不了 L2 里的数据，都会把内循环省下的时间吐回去。raster swizzle、persistent kernel、Stream-K 是 CUTLASS 的三件工具。
# C3 · Tile 调度：wave quantization、raster swizzle、persistent、Stream-K

## 题面速览

- **在系统中的位置**：CUTLASS sm90 的 [[cutlass:include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp|sm90_tile_scheduler.hpp]]（persistent 静态轮转 + raster order / swizzle，参数在 [[cutlass:include/cutlass/gemm/kernel/tile_scheduler_params.h|tile_scheduler_params.h]] 的 `max_swizzle_size`）和 [[cutlass:include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp|sm90_tile_scheduler_stream_k.hpp]]（`DecompositionMode`：DataParallel / SplitK / StreamK / Heuristic）。Blackwell 加了硬件的 Cluster Launch Control（[[cutlass:include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp@issue_clc_query|issue_clc_query]]）：运行中的 CTA“取消”一个还没启动的 CTA、接手它的坐标。
- **你要实现**：`wave_stats`、`raster_order`、`wave_unique_bytes`、`persistent_static`、`stream_k`、`fixup_tiles`、`makespan`。
- **难点在这几条规则**：
  1. 一波 = SM 数 × 每 SM 的 CTA 数；最后一波不满就是浪费。
  2. 条带化的 raster：m 方向每 swizzle 行一条，条带内**按列走**，一波 tile 变成矮胖的方块。
  3. Stream-K：把 `tiles × k_iters` 个 MAC 迭代按 SM 平分，区间在 tile 边界处切开。
  4. 被多个 SM 分段算的 tile 需要 fixup（部分和归约）。

**验收要点**：H100 上 200 个 tile 是 2 波、利用率 76% · 整除与多 CTA · 行优先 · swizzle = 2 的顺序 · raster 是置换 · swizzle 参数校验 · swizzle 减少 L2 流量 · 静态轮转 · Stream-K 小例子与 fixup · 每个 (tile, k) 恰好一次且负载差 ≤ 1 · Stream-K 比 data-parallel 快 · 空输入。

## 直觉与类比

::: callout analogy 食堂打饭
170 个窗口（SM），每份饭（tile）一个窗口打。64 份饭：106 个窗口闲着（5090 上 1024³ 的 GEMM，128×128 tile，只有 37.6% 的利用率——G1 的 bench 里 1024 那一行只有 31 TF/s 就是这个原因）。**persistent**：窗口不关，打完一份接着打下一份，省掉每份开关窗口的时间。**Stream-K**：把每份饭拆成 64 勺，所有勺子平均分给窗口，一份饭可能由两个窗口各打一半，最后要合到一个碗里（fixup）。**raster swizzle**：同一时间打的饭尽量用同几锅菜（A 行块、B 列块），后厨（L2）不用同时备 132 锅。
:::

## 图解：同一个 GEMM 的三种分法

::: flow 200 个 tile × 64 个 k 迭代，132 个 SM
cfg cw=220 rh=100 nw=176
n dp 0 0 "data-parallel" "2 波：128 次迭代" k=todo
n pers 1 0 "persistent 静态" "仍是 2 轮，省启动" k=warn
n sk 2 0 "Stream-K" "每 SM 97 次迭代" k=pass
n fix 2 1 "fixup" "切开的 tile 归约" k=vio
n hy 3 0 "hybrid（CUTLASS 默认）" "整波 DP + 尾波 SK" k=teal
e dp -> pers
e pers -> sk
e sk -> fix
e sk -> hy
cap: 纯 Stream-K 把几乎所有 tile 都切开了（本例 128 个 tile 需要 fixup）；CUTLASS 的做法是整波用 data-parallel，只把最后不满的一两波交给 Stream-K。
:::

## 逐步演示

::: steps 3 个 tile × 4 个 k 迭代，2 个 SM
::: flow
cfg cw=200 rh=90 nw=166
n g 0 0 "全局迭代 0..11" "g → (g/4, g%4)" k=dark
n s0 1 0 "SM 0：0..5" "tile 0 + tile 1 前半" k=acc
n s1 2 0 "SM 1：6..11" "tile 1 后半 + tile 2" k=pass
n f 1.5 1 "tile 1 fixup" "两段部分和相加" k=vio
e g -> s0
e s0 -> s1
e s0:b -> f:l
e s1:b -> f:r
:::
--- step 平分 {hl=g}
T = 3·4 = 12，SM s 负责 `[s·12/2, (s+1)·12/2)`：SM 0 是 0..5，SM 1 是 6..11。
--- step 按 tile 边界切开 {hl=s0,s1}
SM 0：`(0, 0, 4)`（tile 0 的全部 4 步）、`(1, 0, 2)`（tile 1 的前 2 步）；SM 1：`(1, 2, 4)`、`(2, 0, 4)`。
--- step fixup {hl=f}
`fixup_tiles` = `{1: 2}`：tile 1 由 2 个 SM 各算一段，部分和写到 workspace，再由其中一个读回来相加。`makespan` = 6，两个 SM 完全均衡。
--- step 放大到 H100 {hl=s0,s1,f}
200 个 tile × 64 步、132 个 SM：data-parallel 要 2 波 = 128 步；Stream-K 每个 SM 96 或 97 步，`makespan` = 97，快 24%（不计 fixup 开销）。
:::

raster swizzle 的效果（32×32 个 tile，每波 132 个，按“每波不同的 A 块 + B 块数”计，由 `wave_unique_bytes` 算出）：

| swizzle | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 每次 k 步 L2 读的块数（8 波合计） | 295 | 302 | 309 | 220 | 214 |

注意 2 和 4 反而比 1 略差：一波 132 个 tile 和条带形状对不齐时，一波会横跨两个条带。真实调度器按 SM 数和 cluster 形状来选 swizzle，CUTLASS 里叫 `max_swizzle_size`。

## 易错点

- **最后一波按满波算。** `last_wave` 是剩下的 tile 数，`efficiency = tiles / (waves·slots)`。
- **raster 条带内按行走。** 那就和不 swizzle 一样了；题面规定条带内按列（n）走、每列再走完条带内的行。
- **Stream-K 用 `ceil` 给每个 SM 分。** 会让最后一个 SM 少很多；用 `s·T // n` 的整数切分，负载差 ≤ 1。
- **段的 k 范围不在 tile 边界切开。** 一个段跨两个 tile 是错的（`0 <= b < e <= k` 检查）。
- **fixup 把只被一个 SM 算的 tile 也算进去。** 只统计参与 SM 数 > 1 的。

## 三级提示

::: hint 1 I/O 示例与关键问题
```python
wave_stats(200, 132)        # {'waves': 2, 'last_wave': 68, 'efficiency': 0.757...}
wave_stats(64, 170)         # 5090 上 1024^3：{'waves': 1, 'last_wave': 64, 'efficiency': 0.376...}
raster_order(3, 2, 2)       # [(0,0),(1,0),(0,1),(1,1),(2,0),(2,1)]
stream_k(3, 4, 2)           # [[(0,0,4),(1,0,2)], [(1,2,4),(2,0,4)]]
```

#### 关键问题
1. 5090 上 4096³ 的 bf16 GEMM、128×128 tile，利用率是多少？
2. 为什么条带内按列走能减少 L2 流量？
3. Stream-K 的 fixup 有什么代价？什么时候不值得？
4. persistent kernel 除了“不用重新启动 CTA”，还省了什么？
:::

::: hint 2 思路与常见坑
- `raster_order`：`for m0 in range(0, M, s)`，`rows = range(m0, min(m0+s, M))`，`for n in range(N): for mm in rows`。
- `wave_unique_bytes`：按 `wave` 切片，`len({m}) * a + len({n}) * b`。
- `stream_k`：`g0, g1 = s*T//n, (s+1)*T//n`；while 循环：`tile, k = divmod(g, k_iters)`，`end = min(g1, (tile+1)*k_iters)`。
- `fixup_tiles`：`defaultdict(set)` 记录每个 tile 出现在哪些 SM。
:::

::: hint 3 详解：答案
::: answer 关键问题答案
1. 32×32 = 1024 个 tile，170 个 SM：7 波，最后一波 4 个，利用率 86%（`wave_stats(1024, 170)`）。8192³ 是 96%。
2. 同一波里的 tile 共享的 A 行块和 B 列块越多，同时需要的数据越少，L2 越容易装下、命中越高。一波 132 个 tile 排成一整行要 132 个不同的 B 块；排成 8×16 的方块只要 8 + 16 个。
3. 部分和要写到 global workspace、再读回来（一个 128×128 FP32 tile 是 64 KB），还要一个 flag 做同步；切开的 tile 越多开销越大。tile 数正好是 SM 数的整数倍、或者 K 很小时不值得。
4. 每个 tile 的 prologue（TMEM 分配、barrier 初始化、描述符预取）只做一次；更重要的是 epilogue 可以和下一个 tile 的 mainloop 重叠（B4、C1 的双缓冲）。
:::
:::

## 参考解

::: solution solutions/c3_tile_scheduler.py
{{code: solutions/c3_tile_scheduler.py}}
:::

## 面试官追问 & 对照源码

- **“split-K 和 Stream-K 的区别？”** split-K 把每个 tile 的 K 均分成固定的 s 份（每个 tile 都要归约）；Stream-K 只在负载均衡需要的地方切，切的位置由总迭代数决定。
- **“Cluster Launch Control 解决什么问题？”** 静态 persistent 调度不知道哪些 SM 被别的 kernel 占着（例如 PDL 下上一个 kernel 的尾巴、或通信 kernel）；CLC 让 CTA 动态“偷”还没启动的 CTA 的坐标，既有 persistent 的好处又能适应 SM 数变化。
- **“MoE 的 grouped GEMM 怎么调度？”** 每个 expert 的 M 不同，tile 总数在 kernel 里才知道：调度器遍历 `problem_shape` 数组，把所有 group 的 tile 拉平成一条序列再分给 SM。
- **“FlashAttention 的调度？”** FA3/FA4 也是 persistent + 动态取 tile；causal 时不同 Q 块的工作量差一倍，按“工作量从大到小”的顺序取 tile 来减少尾部不齐。
