lede: FlashAttention 1 → 4 算的是同一件事：Q 块留在片上，K/V 一块块流过，online softmax 递推，S 和 P 从不落 HBM。四个版本的区别，是这个递推怎么映射到每一代的硬件上。
# FlashAttention 1 → 4：同一个算法，为硬件改写四次

## 问题在哪

标准 attention：S = QKᵀ（N×N）写到 HBM，softmax 再读出来写回，P·V 再读一遍。N = 8192、32 个 head、BF16 时 S 一个 batch 就是 4 GB。瓶颈是 HBM 流量，不是 FLOP。

| 版本 | 硬件 | 循环顺序 | MMA | 关键想法 | 结果 |
|---|---|---|---|---|---|
| FA1（2022） | A100 / T4 | 外层 K/V、内层 Q：O 每步回 HBM | mma.sync（手写） | 分块 + online softmax | A100 峰值 25–40% |
| FA2（2023） | A100 | 外层 Q（一个 CTA 一块）、内层 K/V：O 留在寄存器 | mma.sync（CuTe） | 换循环顺序、按序列并行、warp 按行分 | A100 上 50–73% |
| FA3（2024） | H100 | 同 FA2 | wgmma | warp specialization、warpgroup pingpong、warpgroup 内重叠、FP8 | H100 上 75%（740 TF FP16） |
| FA4（2025–26） | B200 | 同 FA2，每 CTA 两个 Q tile | tcgen05，S / P / O 在 TMEM | 部分 exp2 走 FMA 多项式；条件 rescale | B200 上 71%（1,613 TF BF16） |

## 解决办法：online softmax + 分块

::: flow 一个 Q 块的一生（FA2 起）
cfg cw=200 rh=100 nw=168
n q 0 0 "Q 块进片上" "读一次" k=acc
n kv 1 0 "K/V 块流过" "j = 0, 1, …" k=teal
n s 2 0 "S = QKᵀ" "片上，不落 HBM" k=warn
n sm 3 0 "online softmax" "m、l、α" k=vio
n o 3 1 "O += P V" "最后写一次 + LSE" k=pass
e q -> kv
e kv -> s
e s -> sm
e sm -> o
e o:l -> kv:b
cap: D1 写递推、D2 写分块与 causal 跳块、G3 在 5090 上把它写成 mma.sync 的 kernel。
:::

::: callout analogy 只记一张小卡片的投票统计
统计一场有几十万张选票的投票，但桌上只放得下一小摞（SMEM）。FA 的做法：每次拿一摞票（K/V 块），在一张小卡片上更新“目前最高票数（m）、按最高票折算的总数（l）、加权结果（acc）”；最高票变了就把卡片上的数按比例折算一下（α）。整场统计只需要这张卡片，从不需要把所有票摊开（N×N 的 S）。多个统计员各统计一部分时，每人交一张卡片，按卡片上的“总量”（LSE）加权合并（split-KV）。
:::

## 为什么每一代要重写

关键比值：每对 (query, key) 要 4d 次 MMA FLOP 和 1 次 exp。MUFU（算 exp2 的单元）每 SM 每时钟 16 次，各代不变；Tensor Core 每代翻倍。

| | Tensor FLOP/clk/SM | exp 时间 / MMA 时间（d = 128） | 后果 |
|---|---|---|---|
| A100 | 2048 | 0.25 | softmax 不是瓶颈：FA2 只要把 HBM 流量省下来 |
| H100 | 4096 | 0.5 | softmax 必须和 GEMM 重叠：FA3 的 pingpong |
| B200 | 8192 | 1.0 | 重叠也不够：FA4 把一部分 exp 挪到 FMA，减少 rescale |
| RTX 5090 | 512 | 0.06 | softmax 几乎免费：FA2 的写法足够（G3） |

（由 D3 的 `exp_to_mma_ratio` 算出。）

## FA4 一个 CTA 里的分工

B200 上 S、P、O 都在 TMEM 里：MMA warp 发 `tcgen05.mma` 算 S = QKᵀ 写进 TMEM；softmax warpgroup 用 `tcgen05.ld` 读 S，算 exp2（MUFU + 一部分 FMA 多项式），把 BF16 的 P 写回 TMEM；MMA warp 再用 TMEM 里的 P 做 P·V，累加进 TMEM 里的 O；correction warpgroup 在需要时 rescale O（D3 的条件 rescale 让它大多时候闲着）；epilogue 最后把 O 写出去。每个 CTA 处理两个 Q tile，让 MMA 和 softmax 在两个 tile 之间 pingpong。源码：[[fa:flash_attn/cute/flash_fwd_sm100.py|flash_fwd_sm100.py]]。

## 和练习题的对应关系

| 题目 | 复刻了什么 | 源码 |
|---|---|---|
| D1 | online softmax 递推、(O, LSE) 合并、split-KV | [[fa:csrc/flash_attn/src/softmax.h@softmax_rescale_o|softmax_rescale_o]]、[[fa:flash_attn/cute/flash_fwd_combine.py|flash_fwd_combine.py]] |
| D2 | FA2 的循环顺序、causal 跳块、HBM 流量 | [[fa:csrc/flash_attn/src/flash_fwd_kernel.h@compute_attn_1rowblock|compute_attn_1rowblock]] |
| D3 | FA4 的 ex2 模拟、条件 rescale、MUFU 预算 | [[fa:flash_attn/cute/utils.py@ex2_emulation|ex2_emulation]]、[[fa:flash_attn/cute/softmax.py@update_row_max|update_row_max]] |
| G3 | FA2 前向在 5090 上的 mma.sync 实现 | `solutions/g3_flash_attn.cu` |
| A2 | S 的 C 布局 = P 的 A 布局 | [[fa:csrc/flash_attn/src/utils.h@convert_layout_acc_Aregs|convert_layout_acc_Aregs]] |
| B4 | FA3 的 warpgroup pingpong | [[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp|同样的 pingpong 思路]] |
