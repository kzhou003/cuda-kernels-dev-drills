lede: FlashAttention 1 到 4 都建立在同一个递推上：softmax 不必等看完整行。维护“目前的最大值、指数和、未归一化的输出”，每来一块 key 就乘一个修正因子。同一个代数还告诉你两段 attention 的结果怎么合并——split-KV decode 和 FA 的 combine kernel 都靠它。
# D1 · Online softmax、(O, LSE) 状态与 split-KV 合并

## 题面速览

- **在系统中的位置**：FA2 的 [[fa:csrc/flash_attn/src/softmax.h@softmax_rescale_o|Softmax::softmax_rescale_o]]（每块更新 row_max、row_sum，并把 acc_o 乘 `exp2(old_max − new_max)`）；全被遮住的行用 `max_scaled = max == -INF ? 0 : max`（[[fa:csrc/flash_attn/src/softmax.h#L76|softmax.h L76]]）。合并：SGLang 的 [[sglang:python/sglang/srt/layers/attention/merge_state.py|merge_state.py]]、FA 的 [[fa:flash_attn/cute/flash_fwd_combine.py|flash_fwd_combine.py]]。G3 在 5090 上把这套递推写进寄存器 fragment。
- **你要实现**：`online_softmax_stats`、`online_attention`、`merge_states`、`split_kv_attention`。
- **难点在这几条规则**：
  1. 新最大值出现时，旧的 l 和 acc 都要乘 `exp(m_old − m_new)`。
  2. 一行到目前为止全被遮住（m = −inf）时，`−inf − (−inf)` = NaN：用 0 做基准。
  3. 合并：`lse = logaddexp(lse_a, lse_b)`，权重 `exp(lse_i − lse)`，两段都空时得 (0, −inf)。
  4. split-KV 允许空段（段数 > key 数）。

**验收要点**：分块统计 = 直接计算（多种块大小）· 空 / 全 −inf · 大数不溢出 · 分块 attention = 参考（块大小 1..64）· 全遮住的行 0 / −inf 且无 NaN · 两半合并 = 全量 · 和空段合并不变 · split-KV（1..50 段）= 参考 · 部分段全被遮住的 split-KV。

## 直觉与类比

::: callout analogy 换汇率的记账
你在记一笔外币账：每笔收入都按“当前最高汇率”换算成本币记下来。突然来了一笔汇率更高的钱，你的基准变了——之前记的所有金额都要按新旧汇率之比（`exp(m_old − m_new)`）折算一下。online softmax 就是这样：m 是基准汇率，l 是折算后的总额，acc 是折算后的加权和。两本分别记的账要合并时，各自带着“总额的对数”（LSE），按 `exp(lse_i − lse)` 的比例加权就行——这就是 split-KV 的 merge。
:::

## 图解：一块 key 进来时

::: flow 每个 key 块对一行做的事
cfg cw=200 rh=100 nw=170
n s 0 0 "S = scale·q·Kᵀ" "这块的分数" k=acc
n m 1 0 "m_new = max(m, max S)" "−inf 时基准用 0" k=warn
n a 2 0 "α = exp(m − m_new)" "旧状态的修正因子" k=vio
n p 1 1 "P = exp(S − m_new)" "这块的权重" k=teal
n st 2 1 "l = α·l + ΣP" "acc = α·acc + P·V" k=pass
e s -> m
e m -> a
e m:b -> p:t
e a -> st
e p -> st
cap: 最后 O = acc / l，lse = m + log l。FA2 在 log2 域里做（`exp2`，分数先乘 `scale·log2 e`），因为 GPU 的 MUFU 只有 `ex2`；数学上完全一样。
:::

## 逐步演示

::: steps 一行分数 [1, 3, 2, 6, 0, 5]，块大小 2
::: flow
cfg cw=200 rh=90 nw=166
n b0 0 0 "块 [1, 3]" "m = 3, l = 1.1353" k=acc
n b1 1 0 "块 [2, 6]" "α = 0.0498" k=warn
n b2 2 0 "块 [0, 5]" "α = 1" k=pass
n fin 3 0 "结果" "l = 1.4452" k=teal
e b0 -> b1
e b1 -> b2
e b2 -> fin
:::
--- step 第一块 {hl=b0}
m 从 −inf 变成 3，α = exp(−inf) = 0（旧状态为空），l = e⁻² + e⁰ = 1.1353。
--- step 新的最大值 {hl=b0,b1}
6 > 3：α = exp(3 − 6) = 0.0498。l = 1.1353 · 0.0498 + e⁻⁴ + e⁰ = 1.0748。acc 也乘同一个 α。
--- step 最大值不变 {hl=b2}
α = 1，直接累加：l = 1.0748 + e⁻⁶ + e⁻¹ = 1.4452，和一次性计算 `Σexp(x − 6)` 完全一样。
--- step 合并两段 {hl=fin}
一个 query、4 个 key 分两段算：前一段 (o, lse) = (1.731, 2.313)，后一段 (3.953, 3.049)；`merge_states` → (3.233, 3.440)，和一次算全部 4 个 key 的结果一致。
:::

## 易错点

- **只修正 l 不修正 acc。** 输出会偏向前面的块。
- **`exp(−inf − (−inf))`。** 全被遮住的行（causal 的左上角、padding、split-KV 的空段）会产生 NaN 并在合并时传染给整行（`test_fully_masked_rows_give_zero_and_neg_inf`）。
- **合并用 `exp(lse_a)` 直接加。** lse 可能是 80、100，exp 溢出；要减去合并后的 lse 再取 exp。
- **合并时 −inf 一侧的权重算成 NaN。** `exp(−inf − (−inf))`，同样的坑，要显式判断。
- **O 没归一化就合并。** 本题约定每段的 O 已经归一化，合并公式是加权平均；如果存的是未归一化的 acc，公式不同。

## 三级提示

::: hint 1 I/O 示例与关键问题
```python
online_softmax_stats([1, 3, 2, 6, 0, 5], 2)   # (6.0, 1.4452)
online_softmax_stats([], 4)                   # (-inf, 0.0)
oa, la = online_attention(q, K[:2], V[:2], 1.0, 2)   # ([[1.731]], [2.313])
ob, lb = online_attention(q, K[2:], V[2:], 1.0, 2)   # ([[3.953]], [3.049])
merge_states(oa, la, ob, lb)                          # ([[3.233]], [3.440])
```

#### 关键问题
1. 为什么 α 要同时乘到 l 和 acc 上？
2. 合并两段时，为什么只需要 lse，不需要分别的 m 和 l？
3. 为什么 decode 要 split-KV，prefill 一般不用？
4. FA2 为什么在 log2 域里算？
:::

::: hint 2 思路与常见坑
- 状态：`m = full(nq, -inf)`、`l = zeros(nq)`、`acc = zeros((nq, dv))`。
- 每块：`S`（加 mask）、`m_new = maximum(m, S.max(1, initial=-inf))`、`safe = where(isfinite(m_new), m_new, 0)`、`alpha = exp(where(isfinite(m), m - safe, -inf))`、`P = exp(S - safe)`。
- 结束：`ok = l > 0`；O 和 lse 对 ok 的行正常算，其余 0 / −inf。
- `merge_states`：`np.logaddexp`；每边权重对 −inf 显式置 0。
:::

::: hint 3 详解：答案
::: answer 关键问题答案
1. O = acc / l，两者都是“以 m 为基准”的量；基准换了，分子分母要同时换，比值才不变。
2. 一段的贡献可以写成 `exp(lse_i) · O_i`（O_i 已归一化，exp(lse_i) 就是这段的 Σexp）。合并 = 按 exp(lse_i) 加权平均，lse 一个数就够。反向传播也只存 lse（FA2 的 `softmax_lse`）。
3. decode 时 query 只有 1 个 token：按 (batch, head, query 块) 并行的 CTA 数太少，喂不饱 SM；切开 key 让多个 CTA 各算一段再合并（FlashDecoding）。prefill 的 query 块已经够多。
4. GPU 的特殊函数单元只有 `ex2`（2 的幂），`exp(x)` 本来就要先乘 log2 e。把 `scale · log2 e` 合进分数的缩放里，省掉每个元素一次乘法（FA2 `softmax_scale_log2`）。D3 的 FA4 更进一步，把部分 `ex2` 挪到 FMA 上。
:::
:::

## 参考解

::: solution solutions/d1_online_softmax.py
{{code: solutions/d1_online_softmax.py}}
:::

## 面试官追问 & 对照源码

- **“FA 的反向为什么只存 lse 就够？”** 反向要重算 P = exp(S − lse)：有了 lse，不需要 m 和 l。
- **“cascade attention / 共享前缀怎么用 merge？”** 共享前缀的 KV 算一次（对所有请求），各请求的私有后缀各算一次，再按 lse 合并——FlashInfer 的 cascade、SGLang 的 [[sglang:sgl-kernel/csrc/attention/cascade.cu|cascade.cu]]。
- **“merge 能用 BF16 存 O 吗？”** O 用 BF16 存问题不大；lse 必须 FP32，否则权重误差直接放大到输出。
- **“为什么 split 的段数不能太多？”** 每段都要写一份 (O, lse) 到 HBM 再读回来合并，段数 × head_dim 的额外流量；FA3 / FlashInfer 用启发式按 SM 数和序列长度选段数。
