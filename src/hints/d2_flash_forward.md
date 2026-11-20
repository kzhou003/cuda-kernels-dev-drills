lede: FA1 和 FA2 算的是同一个东西，区别在循环顺序：FA1 外层扫 K/V，每一步都把所有 Q 块的 O 从 HBM 读出来再写回去；FA2 外层是 Q 块（一个 CTA 一块），O 一直留在寄存器里。再加上 causal 跳过对角线上方的块，几乎省一半计算。
# D2 · FlashAttention 前向分块：循环顺序、causal 跳块与 HBM 流量

## 题面速览

- **在系统中的位置**：FA2 的 [[fa:csrc/flash_attn/src/flash_fwd_kernel.h@compute_attn_1rowblock|compute_attn_1rowblock]]：一个 CTA 处理一个 (batch, head, Q 块)，从 `n_block_max` 往回扫 K/V 块（[[fa:csrc/flash_attn/src/flash_fwd_kernel.h#L93-L97|n_block_max 的计算]]，causal 时截到对角线），只有跨对角线的块走带 mask 的路径。G3 在 5090 上写的就是这个 kernel 的简化版。
- **你要实现**：`flash_forward(Q, K, V, scale, Br, Bc, causal)`（返回 O、LSE 和计算 / 跳过的块数）、`causal_blocks`、`hbm_bytes`（FA1 vs FA2 的流量公式）。
- **难点在这几条规则**：
  1. causal 用**右下对齐**：query i 看得见 key j ⇔ `j ≤ i + (nk − nq)`。
  2. 整块都看不见（块内最“左下”的元素都看不见）→ 跳过；整块都看得见 → 不用 mask；否则逐元素 mask。
  3. nq > nk 时前面若干行一个 key 都看不见：输出 0、lse = −inf。
  4. FA1：Q 读 Tc 次，O 读 Tc 次写 Tc 次；FA2：K、V 各读 Tr 次，Q、O 各一次。

**验收要点**：多种形状 × causal / 非 causal = 参考 · 非 causal 计算全部块 · 64×64、16×16 块的 causal 计算 10 个跳 6 个 · nq > nk 的整行遮住 · `causal_blocks` = 实际计算块数 · 长序列 causal 约省一半 · 流量公式 · FA2 流量小于 FA1。

## 直觉与类比

::: callout analogy 批改作业的两种顺序
老师（SM）要给 N 个学生（Q 块）批 N 道题（K/V 块）。FA1 的顺序：拿起第 1 道题，把所有学生的卷子从柜子（HBM）里拿出来，批完这道再放回去；再拿第 2 道题，又把所有卷子拿出来……卷子进出柜子 N 次。FA2 的顺序：每个老师负责一个学生，把这个学生的卷子放在桌上（寄存器），一道一道题批完，最后放回柜子一次；题目（K/V）每个老师都要看一遍，但题目大家都在看，大多在 L2 里。causal 就是“第 i 个学生只做前 i 道题”：后面的题根本不用拿出来。
:::

## 图解：一个 CTA 里发生的事

::: flow FA2：Q 块在外层，O 留在片上
cfg cw=200 rh=100 nw=168
n q 0 0 "Q 块 i" "读一次，留在片上" k=acc
n kv 1 0 "K/V 块 j" "j ≤ 对角线" k=teal
n s 2 0 "S = QKᵀ" "对角线块才 mask" k=warn
n sm 3 0 "online softmax" "D1 的递推" k=vio
n o 3 1 "O 累加器" "最后写一次 + LSE" k=pass
n skip 1 1 "对角线上方" "根本不进循环" k=ghost
e q -> kv
e kv -> s
e s -> sm
e sm -> o
e kv:b -> skip:t
cap: 外层（Q 块）是并行的：每个 CTA 一块，长序列时 CTA 数 = batch × head × N/Br，足够喂饱所有 SM。FA1 外层是 K/V，只能按 batch × head 并行，batch 小、序列长时 SM 吃不满——这是 FA2 的第二个改进。
:::

## 逐步演示

::: steps N = 256，块 64×64，causal
::: flow
cfg cw=180 rh=90 nw=150
n r0 0 0 "Q 块 0" "算 1 块" k=acc
n r1 1 0 "Q 块 1" "算 2 块" k=acc
n r2 2 0 "Q 块 2" "算 3 块" k=acc
n r3 3 0 "Q 块 3" "算 4 块" k=acc
n d 1.5 1 "对角线块" "4 个，要 mask" k=warn
e r0 -> r1
e r1 -> r2
e r2 -> r3
:::
--- step 块网格 {hl=r0,r1,r2,r3}
4×4 = 16 个 (Q 块, K 块) 对。causal 下第 i 个 Q 块只需要 K 块 0..i：1 + 2 + 3 + 4 = 10 个，跳过 6 个（`flash_forward(...)[2]` = `{'computed': 10, 'skipped': 6}`）。
--- step 哪些要 mask {hl=d}
对角线上的 4 个块一半可见一半不可见，逐元素 mask；其余 6 个计算的块完全可见，走不带 mask 的快路径。
--- step 长序列 {hl=r3}
N = 4096、块 128：32×32 = 1024 个块里算 528 个，约 51.6%（`causal_blocks(4096, 4096, 128, 128)`）。
--- step HBM 流量 {hl=r0,r3}
d = 128、块 128、BF16（`hbm_bytes`）：N = 4096 时 FA1 98 MiB、FA2 66 MiB；N = 16384 时 1544 vs 1032 MiB。FA2 里 K/V 的 Tr 次重复读大多命中 L2，实际 HBM 流量更接近 4·N·d·2 字节。
:::

## 易错点

- **causal 用左上对齐。** nq ≠ nk（例如 decode 或 chunked prefill）时，右下对齐才对：最后一个 query 能看到所有 key。
- **跳块判断用了块的左上角。** 要用“最左下”的元素 (i1−1, j0) 判断整块是否不可见。
- **整行不可见时出 NaN。** nq > nk 的 causal 会出现（`test_causal_with_more_queries_than_keys_masks_whole_rows`）。
- **`causal_blocks` 和实际计算不一致。** 尾块（N 不整除块大小）要用 `min(i0+Br, nq)`。
- **FA1 公式漏了 O 的读。** FA1 每一轮都要把 O 读出来更新，所以是 3·Tc（Q 读、O 读、O 写）。

## 三级提示

::: hint 1 I/O 示例与关键问题
```python
O, lse, st = flash_forward(Q, K, V, 0.25, 16, 16, causal=True)   # Q,K,V: 64x16
st                                    # {'computed': 10, 'skipped': 6}
causal_blocks(4096, 4096, 128, 128)   # 528
hbm_bytes(4096, 128, 128, 128, "fa1") / 2**20   # 98.0
hbm_bytes(4096, 128, 128, 128, "fa2") / 2**20   # 66.0
```

#### 关键问题
1. 一个 (Q 块, K 块) 什么时候可以整块跳过？什么时候可以不加 mask？
2. FA1 里 O 为什么要反复读写 HBM？FA2 为什么不用？
3. FA2 的 K/V 被读 Tr 次，为什么实际不慢？
4. causal 时不同 Q 块的工作量差多少？对调度有什么影响？
:::

::: hint 2 思路与常见坑
- 外层 `for i0 in range(0, nq, Br)`：每块初始化 `m, l, acc`。
- 内层 `for j0 in range(0, nk, Bc)`：先判断跳过（计数），再算 S，判断是否需要 mask（用 `np.where(jj <= ii + off, S, -inf)`），然后是 D1 的递推。
- 写回 O、lse 时处理 `l == 0` 的行。
- `hbm_bytes`：`Tr = ceil(N/Br)`、`Tc = ceil(N/Bc)`，两个公式各一行。
:::

::: hint 3 详解：答案
::: answer 关键问题答案
1. 块内最左下的元素 (i1−1, j0) 都看不见（`j0 > i1 − 1 + off`）就跳过；最右上的元素 (i0, j1−1) 都看得见（`j1 − 1 ≤ i0 + off`）就不用 mask。
2. FA1 外层是 K/V：处理完一块 K/V 时，每个 Q 块的 O 都只是“部分结果”，片上放不下所有 Q 块的 O，只能写回 HBM、下一轮再读。FA2 外层是 Q：一个 Q 块的 O 从头到尾留在寄存器里，最后写一次。
3. 同一时刻很多 CTA（同一个 head 的不同 Q 块）在读同样的 K/V 块，L2 命中率很高；HBM 实际只读一遍左右。
4. 第 i 个 Q 块要算 i+1 个 K 块，最后一块是第一块的 N/Br 倍。如果按 Q 块顺序发 CTA，最后一波全是重活，尾部不齐。FA2 / FA3 会把重的 Q 块先发（或者 persistent 调度按工作量排序，C3）。
:::
:::

## 参考解

::: solution solutions/d2_flash_forward.py
{{code: solutions/d2_flash_forward.py}}
:::

## 面试官追问 & 对照源码

- **“FA2 里一个 CTA 的 4 个 warp 怎么分工？”** 按 Q 的行分：每个 warp 16 行（或更多），各自看全部 K 列，不需要 warp 之间归约 softmax。FA1 是按 K 分（split-K 式），每块都要在 warp 之间归约 max 和 sum，慢。G3 用的就是 FA2 的分法。
- **“FA3 在这个结构上加了什么？”** TMA 加载 K/V（B3）、producer / consumer warp specialization（B4）、两个 warpgroup 的 pingpong 让 softmax 藏在另一个 warpgroup 的 GEMM 后面、以及 FP8。
- **“为什么 FA2 对 head_dim 256 慢？”** O 累加器 = Br × d 个 FP32 寄存器；d 大时只能减小 Br，强度下降（A1）。
- **“sliding window / local attention 怎么跳块？”** 同样的思路，只是可见区域是一条带：块的上下界都要截（`n_block_min` 和 `n_block_max`）。
