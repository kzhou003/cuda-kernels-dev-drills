lede: Ampere 让“搬运”变成异步：线程发出 cp.async 就去算别的，搬运按“组”提交、按“除最新 N 组外都到齐”来等。多级 ring buffer 把一块数据的几百个周期延迟藏在前面几块的计算后面。
# B1 · cp.async 多级流水：commit / wait_group 与 K 循环时间线

## 题面速览

- **在系统中的位置**：CUTLASS sm80 的主循环 [[cutlass:include/cutlass/gemm/collective/sm80_mma_multistage.hpp|sm80_mma_multistage.hpp]]：prologue 先发 Stages−1 块，之后每轮 `cp_async_wait<Stages-2>()` → `__syncthreads()` → 发下一块 → 算。PTX 包装在 [[cutlass:include/cute/arch/copy_sm80.hpp@cp_async_fence|cp_async_fence]]（commit_group）和 [[cutlass:include/cute/arch/copy_sm80.hpp@cp_async_wait|cp_async_wait]]。G2 在 5090 上写的就是这段。
- **你要实现**：`CpAsyncQueue`（一个线程的 cp.async 组语义）、`multistage_ops(T, S)`（主循环的操作序列）、`simulate(T, S, L, C)`（假时钟上的时间线）。
- **难点在这几条规则**：
  1. `wait_group n` 等的是**已提交的组**：除最新 n 组外全部完成；没 commit 的拷贝不算。
  2. **空组也要 commit**：尾部没有新块可发时照样 commit，否则 `wait<S-2>` 等的不再是你以为的那一块。
  3. 新块写进 slot `(k+S-1) % S = (k-1) % S`，也就是上一轮刚算完的那块。
  4. 稳态周期 ≈ `max(C, L / (S-1))`。

**验收要点**：wait_group 留下最新 n 组 · 未提交的不等 · 空组也算组 · n < 0 报错 · S=3、T=4 的精确操作序列 · S=1 同步 · 每次 mma 读的块都已到齐 · 不覆盖还要用的 slot · 三组精确时间线 · stage 越多不会更慢。

## 直觉与类比

::: callout analogy 回转寿司的传送带
厨房（HBM）做一盘寿司要 L 分钟才能送到你面前，你吃一盘要 C 分钟。同步的吃法：点一盘、干等、吃完再点——大部分时间在等。cp.async 多级流水是**传送带上同时放 S−1 盘**：坐下时先点 S−1 盘（prologue），之后每吃完一盘就再点一盘（发下一块），保证传送带上一直有 S−1 盘在路上。只要 (S−1)·C ≥ L，你永远不用等。`commit_group` 是“这几样算一单”，`wait_group n` 是“除了最近点的 n 单，前面的都上齐了再开吃”。最后几轮没东西可点了也要“报一张空单”，不然你数单子的时候会数错。
:::

## 图解：一个 stage 的生命周期

::: flow 4 级 ring buffer 上的 slot
cfg cw=210 rh=96 nw=170
n issue 0 0 "cp.async 发出" "slot (k+3) % 4" k=acc
n fly 1 0 "在路上" "约 L 个周期" k=ghost
n wait 2 0 "wait_group 2" "+ __syncthreads" k=warn
n mma 3 0 "ldmatrix + mma" "读 slot k % 4" k=pass
n free 1.5 1 "slot 空出" "下一轮被覆盖" k=teal
e issue -> fly
e fly -> wait
e wait -> mma
e mma:b -> free:r
e free:l -> issue:b
cap: 一个 slot 被重新写之前，必须所有 warp 都读完它——这就是 wait 之后那次 `__syncthreads()` 的另一半作用（前一半是让所有线程的拷贝都可见）。Hopper 把这两件事换成了 mbarrier（B2）。
:::

## 逐步演示

::: steps T = 4 个 K 块、S = 3 级
::: flow
cfg cw=190 rh=90 nw=160
n pro 0 0 "prologue" "load 0,1" k=acc
n k0 1 0 "k = 0" "wait 1 → {0}" k=pass
n k1 2 0 "k = 1" "wait 1 → {0,1}" k=pass
n k2 3 0 "k = 2、3" "空组 commit" k=warn
e pro -> k0
e k0 -> k1
e k1 -> k2
:::
--- step prologue {hl=pro}
`load 0, commit, load 1, commit`：2 个组在路上。
--- step k = 0 {hl=pro,k0}
已提交 2 组，`wait 1` → 最老的 1 组完成 → 驻留 `{0}`。然后 `load 2, commit`，`mma 0`。
--- step k = 1 {hl=k0,k1}
已提交 3 组（去掉已等完的），`wait 1` → 驻留 `{0, 1}`。`load 3, commit`，`mma 1`。
--- step k = 2 {hl=k1,k2}
没有新块可发（2 + 2 = 4 ≥ T），但**照样 commit 一个空组**。`wait 1` → `{0, 1, 2}`，`mma 2`。
--- step k = 3 {hl=k2}
`wait 1` → `{0, 1, 2, 3}`：多亏了 k = 2 时的空组，“除最新 1 组外”里包含了块 3 那一组。少了空组，这里块 3 不保证到齐。
:::

时间线（L = 100，C = 40，T = 8，由 `simulate` 算出）：

| S | 总周期 | mma 开始时刻 | 利用率 |
|---|---|---|---|
| 1（同步） | 1120 | 100, 240, 380, … | 0.29 |
| 2 | 840 | 100, 200, 300, … | 0.38 |
| 3 | 480 | 100, 140, 200, 240, … | 0.67 |
| 4 | 420 | 100, 140, 180, … | 0.76 |
| 5 | 420 | 同上 | 0.76 |

S = 4 时 (S−1)·C = 120 ≥ L = 100，延迟完全藏住，之后再加 stage 没有用（只会多占 SMEM）。S = 3 时 80 < 100，每两轮要等一次。

## 易错点

- **`wait_group` 也等了未提交的拷贝。** 真实硬件不会；忘了 commit 的拷贝永远“不保证到齐”（`test_uncommitted_copies_are_not_waited`）。
- **尾部不提交空组。** `test_every_mma_reads_a_resident_tile` 在 T < S 和 T ≫ S 的多种组合下逐个检查。
- **用 `wait_group 0`。** 正确，但把流水变成了同步：每轮都等最新的那块，延迟没藏住。
- **新块写进 `k % S`。** 那是本轮要算的块；`test_loads_never_overwrite_a_slot_still_needed` 抓这个。
- **wait 没推进时钟，或推进到了错误的组。** 要用“需要完成的那些组里最晚落地的时刻”。

## 三级提示

::: hint 1 I/O 示例与关键问题
```python
q = CpAsyncQueue()
for t in range(3): q.cp_async(t); q.commit_group()
q.wait_group(2)          # {0}
q.wait_group(0)          # {0, 1, 2}
multistage_ops(4, 3)[:8] # [('load',0),('commit',),('load',1),('commit',),('wait',1),('load',2),('commit',),('mma',0)]
simulate(8, 4, 100, 40)  # {'total': 420, 'mma_start': [100, 140, ...], 'util': 0.76}
```

#### 关键问题
1. 第 k 轮 wait 之前一共提交了几组？`wait S-2` 之后哪些块一定到齐？
2. 为什么是 S−2 而不是 S−1？
3. 稳态下一轮多长？什么时候再加 stage 没用？
4. 真实 kernel 里 wait 之后为什么还要 `__syncthreads()`？
:::

::: hint 2 思路与常见坑
- 队列：`_open`（当前组）、`_groups`（已提交未确认的组，老→新）、`resident`。`wait_group(n)`：`done = groups[:-n] if n else groups`。注意 `[:-0]` 是空列表！
- `multistage_ops`：S == 1 单独处理；否则 prologue `range(S-1)`（`s < T` 才 load，总是 commit），主循环 `wait S-2 → (load) → commit → mma`。
- `simulate`：同样遍历 ops；用列表记录“当前未提交组的落地时刻”，commit 时整体推进 `groups`，wait 时取 `max`。
:::

::: hint 3 详解：答案
::: answer 关键问题答案
1. (S−1) + k 组（prologue S−1 组 + 前 k 轮每轮 1 组）。`wait S-2` 后前 k+1 组完成，也就是块 0..k。
2. 第 k 轮要用的是第 k+1 组（从 1 数）。总共 (S−1)+k 组，留下最新 S−2 组，正好前 k+1 组完成。用 S−1 只保证块 0..k−1。
3. `max(C, L/(S−1))`；当 (S−1)·C ≥ L 时已是 C，再加 stage 只会多占 SMEM、降低 occupancy——G2 在 5090 上就测得 2 级比 3、4 级快。
4. 两个原因：(a) `wait_group` 只保证**本线程**发的拷贝到齐，别的线程搬的那部分要靠 barrier 才能看到；(b) 保证所有 warp 都读完了上一轮的 slot，才能覆盖它。
:::
:::

## 参考解

::: solution solutions/b1_cp_async_pipeline.py
{{code: solutions/b1_cp_async_pipeline.py}}
:::

## 面试官追问 & 对照源码

- **“cp.async.ca 和 .cg 有什么区别？”** `.ca` 在 L1 缓存，`.cg` 只经 L2（绕过 L1）；16 B 拷贝通常用 `.cg`。zero-fill 版本（src-size = 0）用于越界：[[cutlass:include/cute/arch/copy_sm80.hpp@SM80_CP_ASYNC_CACHEGLOBAL_ZFILL|SM80_CP_ASYNC_CACHEGLOBAL_ZFILL]]。
- **“为什么 Hopper 不用 cp.async 了？”** H100 每 SM 每时钟要 4096 FLOP，每个线程算地址、发 16 B 拷贝、再 `__syncthreads`，占掉的线程和指令发射带宽太多。TMA 一个线程一条指令搬整个 tile，完成用 mbarrier 按字节记账（B2、B3）。
- **“FlashAttention-2 的 K/V 也是这么流水的吗？”** 是：FA2 在 sm80 上用 cp.async 双缓冲 K、V（`flash_fwd_kernel.h` 里的 `cp_async_fence / cp_async_wait`），G3 为了简单用的是同步加载，留作扩展练习。
