lede: Hopper 之后，kernel 里一切异步协作（TMA 写完了没有、MMA 读完了没有）都靠 SMEM 里的 mbarrier：到达计数 + 事务字节 + 一个相位位。理解了相位位怎么翻、等待方的 parity 怎么跟着翻，就读得懂 CUTLASS 的整个 pipeline 层。
# B2 · mbarrier：相位位、事务字节与 TMA 生产者/消费者环

## 题面速览

- **在系统中的位置**：PTX 包装在 [[cutlass:include/cutlass/arch/barrier.h@ClusterTransactionBarrier|ClusterTransactionBarrier]]（`arrive_and_expect_tx`、`wait(phase)` 自旋在 `mbarrier.try_wait.parity` 上）；环形缓冲在 [[cutlass:include/cutlass/pipeline/sm90_pipeline.hpp@PipelineTmaAsync|PipelineTmaAsync]]，位置是 [[cutlass:include/cutlass/pipeline/sm90_pipeline.hpp@PipelineState|PipelineState]]，producer 从 [[cutlass:include/cutlass/pipeline/sm90_pipeline.hpp@make_producer_start_state|make_producer_start_state]]（phase = 1）开始。Blackwell 的 [[cutlass:include/cutlass/pipeline/sm100_pipeline.hpp@PipelineUmmaAsync|PipelineUmmaAsync]] 是同一套东西。
- **你要实现**：`MBarrier`、`PipelineState`、`make_producer_start_state`、`TmaPipeline`（每个 stage 一对 full / empty barrier）。
- **难点在这几条规则**：
  1. 相位完成 = **pending arrive 归零且 tx 字节归零**；完成时硬件自己把计数装回初值、翻转相位位。
  2. `try_wait(parity)` ⇔ `parity != phase`：等的是“parity 指的那个相位已经完成”。
  3. producer 先 `arrive_and_expect_tx`（**先记字节再到达**），否则相位会在一个字节都没写时完成。
  4. `PipelineState` 每绕一圈翻转 phase；producer 从 phase 1 开始，所以第一圈不等。

**验收要点**：到齐才翻相位 · 自动重置 · 字节挡住相位 · complete_tx 可以先到 · 先 arrive 后 expect 会提前完成（反例）· state 绕圈翻转 · producer 第一圈不等、第二圈要等 · 失败的 acquire 不改状态 · 随机交错下 1–5 级、1–3 个 consumer 都按顺序收到每一块且从不读到被覆盖的数据。

## 直觉与类比

::: callout analogy 只有一位的“第几轮”灯
每个 stage 门口有两盏灯：“满”（full）和“空”（empty）。每盏灯只有一个比特：亮 / 灭，每完成一轮就切换一次。等的人手里拿着一张卡片写着“我在等 parity = p 的那一轮完成”，看到灯的状态不等于 p 就进门（`p != phase`）。因为只有一位，等的人最多只能落后一轮——ring buffer 正好保证这一点：producer 不可能比 consumer 快两圈，因为它要等“空”灯。“满”灯的完成条件比较特别：producer 到了（1 次 arrive）**而且**快递员（TMA）报告的字节数和预约的一样多。producer 必须先预约字节再打卡，不然打卡的瞬间灯就切换了，consumer 冲进去拿到的是空箱子。
:::

## 图解：一个 stage 的两盏灯

::: flow full / empty 的握手
cfg cw=220 rh=104 nw=176
n pa 0 0 "producer acquire" "等 empty 翻相位" k=acc
n ex 1 0 "arrive_and_expect_tx" "full：+tx，arrive 1" k=acc
n tma 2 0 "TMA 写 SMEM" "complete_tx(bytes)" k=dark
n cw 2 1 "consumer wait" "等 full（parity=c_phase）" k=pass
n rel 1 1 "consumer release" "empty：arrive" k=pass
e pa -> ex
e ex -> tma
e tma:b -> cw:t "full 翻相位"
e cw -> rel
e rel:l -> pa:b "empty 翻相位"
cap: full = MBarrier(1)：1 次到达 + 事务字节；empty = MBarrier(consumer 数)。CUTLASS 里 TMA 的 `complete_tx` 是硬件做的（`cp.async.bulk.tensor … .mbarrier::complete_tx::bytes`），本题由 `tma_complete` 模拟。
:::

## 逐步演示

::: steps 2 级、1 个 consumer、每块 64 字节
::: flow
cfg cw=200 rh=92 nw=168
n p 0 0 "producer" "state (idx, phase)" k=acc
n f0 1 0 "full[0]" "phase 位" k=teal
n e0 1 1 "empty[0]" "phase 位" k=warn
n c 2 0 "consumer" "state (idx, phase)" k=pass
e p -> f0
e f0 -> c
e c:b -> e0:r
e e0:l -> p:b
:::
--- step 第一圈不用等 {hl=p,e0}
producer 状态 (0, 1)：`empty[0].try_wait(1)`，empty 的 phase 是 0 ≠ 1 → 真。对 full[0] 登记 64 字节并到达：pending = 0，但 tx = 64，**相位不翻**。同样拿到 slot 1。
--- step 第二圈要等 {hl=p,e0}
producer 状态 (0, 0)：`empty[0].try_wait(0)`，phase 仍是 0 → 假，`producer_try_acquire` 返回 False，**什么都不改**。
--- step 字节一半一半到 {hl=f0}
consumer 状态 (0, 0)：`full[0].try_wait(0)` → 假。TMA 写了 32 字节：tx = 32，还是假。又写了 32：tx = 0 且 pending = 0 → full[0] 翻到 phase 1，consumer 的 `try_wait(0)` 变真。
--- step 释放 {hl=c,e0,p}
consumer 读完、`consumer_release`：empty[0] 的 1 次到达凑齐 → phase 1。producer 的 `try_wait(0)` 终于为真，拿到 slot 0 开始第二圈。consumer 前进到 (1, 0)，绕回 slot 0 时会变成 (0, 1)。
:::

## 易错点

- **parity 判断写反。** 写成 `parity == phase` 会让 producer 第一圈就卡死、consumer 读到空数据；随机交错测试会报 deadlock 或 “stale data”。
- **相位完成后不重置计数。** 第二圈 pending 已经是 0，一次 arrive 就变成 −1，永远完成不了（`test_barrier_resets_itself_for_next_phase`）。
- **producer 先 arrive 后 expect。** `test_arrive_before_expect_would_complete_early` 演示了这个反例：相位会立刻完成。
- **acquire 失败时也登记了字节。** 下次成功时又登记一次，full 永远等不到双倍的字节（`test_failed_acquire_changes_nothing`）。
- **producer 从 phase 0 开始。** 第一圈就等 empty，而 empty 还没有人 arrive 过：死锁。

## 三级提示

::: hint 1 I/O 示例与关键问题
```python
b = MBarrier(3)
b.try_wait(1), b.try_wait(0)   # (True, False)   初始 phase 0：“上一个相位”算完成
b.arrive(); b.arrive(); b.arrive()
b.phase, b.completed           # (1, 1)
b = MBarrier(1); b.arrive_and_expect_tx(1024); b.complete_tx(512); b.phase   # 0
b.complete_tx(512); b.phase                                                   # 1
```

#### 关键问题
1. 为什么一位相位就够了？什么情况下不够？
2. consumer 的初始 phase 为什么是 0、producer 的为什么是 1？
3. 如果有 2 个 consumer warpgroup，empty barrier 的到达数应该是多少？
4. TMA 拷贝的是越界的尾 tile，expect_tx 写多少？
:::

::: hint 2 思路与常见坑
- `MBarrier` 状态：`expected, _pending, _tx, phase, completed`。所有会改计数的方法最后都调用一个 `_maybe_complete()`：`pending == 0 and tx == 0` 时翻转、计数、重置。
- `expect_tx` 只加字节、不检查完成（此时 pending 一般还没到 0）。
- `PipelineState.advance`：`index += 1`，等于 stages 时归零并 `phase ^= 1`。
- `TmaPipeline.producer_try_acquire`：先 `empty[i].try_wait(state.phase)`，失败直接返回 False；成功才 `full[i].arrive_and_expect_tx(tx_bytes)`。
:::

::: hint 3 详解：答案
::: answer 关键问题答案
1. 等待方只需要区分“我等的这一轮完成了没有”和“还在上一轮”。只要等待方永远不落后两轮就够；ring buffer 里 producer 要先等 empty、consumer 要先等 full，两边互相牵制，差距不会超过一圈。
2. consumer 第一圈要等 full 的第 0 个相位完成：parity 0。producer 第一圈不应该等（缓冲本来就空），用 parity 1 指向“上一个相位”，初始状态下它算已完成。
3. 2（或者 2 × 每个 warpgroup 里实际调用 arrive 的线程数，CUTLASS 里是每个 warpgroup 一个 elected 线程，所以 `num_consumers` = warpgroup 数）。见 [[cutlass:examples/cute/tutorial/hopper/wgmma_tma_sm90.cu|wgmma_tma_sm90.cu]] 里 `ConsumerBarType::init(&consumer_mbar[pipe], 128)`——那个教程里是每个线程都 arrive，所以是 128。
4. 整个 box 的字节数，越界填零的部分也算（B3 的 `tx_bytes`）。
:::
:::

## 参考解

::: solution solutions/b2_mbarrier_pipeline.py
{{code: solutions/b2_mbarrier_pipeline.py}}
:::

## 面试官追问 & 对照源码

- **“try_wait 和 test_wait 有什么区别？”** `test_wait` 立即返回；`try_wait` 可以在硬件里挂起一段时间（带超时提示），避免纯自旋占发射槽。CUTLASS 的 [[cutlass:include/cutlass/arch/barrier.h@ClusterBarrier|ClusterBarrier]]`::wait` 就是 try_wait 的循环。
- **“cluster 里别的 CTA 能 arrive 我的 barrier 吗？”** 能：`mbarrier.arrive.shared::cluster` 带目标 CTA id。multicast TMA 写进多个 CTA 时，每个目标 CTA 自己的 full barrier 各收到自己那份字节；而 empty barrier 要收集所有用到这块数据的 CTA 的 release（`consumer_release` 里的 `dst_blockid_`）。
- **“wgmma 是异步的，consumer 什么时候能 release？”** 必须在 `warpgroup_wait<N>` 确认读这块 SMEM 的 wgmma 都完成之后；CUTLASS 的 mainloop 里 release 的是“上一块”，让一个 wgmma 批次保持在飞行中。
- **“Blackwell 上谁来 arrive empty？”** MMA 是一个线程发起的异步 `tcgen05.mma`，用 `tcgen05.commit` 让硬件在 MMA 读完后 arrive 指定的 mbarrier：[[cutlass:include/cutlass/arch/barrier.h@umma_arrive|umma_arrive]]。
