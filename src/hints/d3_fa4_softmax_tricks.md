lede: Tensor Core 每代翻倍，算 exp 的单元（MUFU）却没变：到 B200 上，一次 exp 的时间已经和 128 维的 MMA 一样长。FA4 的回答有两个：把一部分 exp2 挪到 FMA 单元上用多项式算；行最大值只涨一点时干脆不 rescale。
# D3 · FA4 的 softmax：FMA 上的 exp2 多项式与条件 rescale

## 题面速览

- **在系统中的位置**：FA4（CuTe DSL）的 [[fa:flash_attn/cute/utils.py@ex2_emulation|ex2_emulation]]（`fmax(x, −127)`、用 `add.rm` 取 floor、[[fa:flash_attn/cute/utils.py@POLY_EX2|POLY_EX2]] 的 3 次多项式、整数加法拼进指数位）；条件 rescale 在 [[fa:flash_attn/cute/softmax.py@update_row_max|Softmax.update_row_max]]（`rescale_threshold`，MLA 内核默认 8.0）；哪些 fragment 用模拟由 [[fa:flash_attn/cute/config.py|config.py]] 的 `ex2_emu_freq / ex2_emu_start_frg` 决定。**真跑需要 B200（sm100a）**；这两个技巧本身在任何 GPU 上都成立，可以作为 G3 的扩展练习移植到 5090 上。
- **你要实现**：`ex2_emulation`、`softmax_attention_cond_rescale`、`exp_to_mma_ratio`、`balanced_emulation_fraction`。
- **难点在这几条规则**：
  1. `2^x = 2^⌊x⌋ · 2^f`，f ∈ [0, 1)：负数要用 floor（−0.5 = −1 + 0.5），不是截断。
  2. 2^⌊x⌋ 不用乘法：把 j 左移 23 位加到 p 的 IEEE 位模式上（指数位 +j）；x 先夹到 ≥ −127。
  3. 条件 rescale：log2 域里最大值的增量 ≤ threshold 时**保持旧的 m**，P 的值最多到 2^threshold；最后 acc / l 用的是同一个 m，结果仍然精确。
  4. 每对 (q, k)：MMA 4d FLOP、exp 1 次；比值 = `(1/mufu) / (4d/tc)`。

**验收要点**：整数点上精确 · 最大相对误差 < 2e−4 且确实是近似 · 负的小数用 floor · 极小输入夹住 · 三种 threshold 都精确 · threshold 越大 rescale 越少 · 大跳变仍然 rescale 且按行计数 · threshold = 0 时每次涨都 rescale · A100 / H100 / B200 的比值 0.25 / 0.5 / 1 · 平衡比例。

## 直觉与类比

::: callout analogy 只有一台计算器的会计
每算一个注意力分数，会计要做 4·d 次乘加（交给一排算盘 = Tensor Core）和 1 次查指数表（交给唯一一台计算器 = MUFU）。算盘每代翻倍，计算器还是那一台，到 B200 上两边一样慢。办法一：让一部分指数用算盘按公式算（3 次乘加的多项式 + 改一下指数位），两边都不闲着。办法二：换汇率（rescale）本来是“基准变了就把账本全部重抄一遍”，现在改成“基准只涨了一点点就不换”，反正最后除法时分子分母用的是同一个基准，账一样平。
:::

## 图解：一个 FA4 CTA 里的 softmax 路径

::: flow S（TMEM）→ softmax → P（TMEM），O 由 correction warp 修正
cfg cw=200 rh=100 nw=170
n s 0 0 "S = QKᵀ" "tcgen05 → TMEM" k=acc
n mx 1 0 "行最大值" "涨幅 ≤ 8：不换 m" k=warn
n mufu 2 0 "MUFU ex2" "大部分元素" k=vio
n fma 2 1 "FMA 多项式" "一部分元素" k=teal
n p 3 0 "P（BF16）" "写回 TMEM → PV" k=pass
n cor 0 1 "correction warp" "需要时才 rescale O" k=ghost
e s -> mx
e mx -> mufu
e mx:b -> fma:l
e mufu -> p
e fma -> p
e mx:b -> cor:r
cap: 两个技巧都在省 B200 上最紧的资源：MUFU 吞吐（每 SM 每时钟 16 次）和 TMEM 带宽（每次 rescale 都要把 O 读出来、乘、写回）。
:::

## 逐步演示

::: steps 每一代：exp 和 MMA 谁慢（d = 128，MUFU 16 次/时钟/SM）
::: flow
cfg cw=190 rh=90 nw=160
n a 0 0 "A100" "exp / MMA = 0.25" k=pass
n h 1 0 "H100" "0.5：要重叠" k=warn
n b 2 0 "B200" "1.0：重叠也不够" k=todo
n fa4 3 0 "FA4" "57% 挪到 FMA" k=teal
e a -> h
e h -> b
e b -> fa4
:::
--- step A100 {hl=a}
Tensor Core 2048 FLOP/时钟/SM：每对 (q,k) 的 MMA 要 4·128 / 2048 = 0.25 时钟，exp 要 1/16 = 0.0625：比值 0.25（`exp_to_mma_ratio(2048, 16, 128)`）。softmax 不是瓶颈，FA2 不用管它。
--- step H100 {hl=h}
4096：比值 0.5。softmax 占 MMA 时间一半，FA3 用两个 warpgroup 的 pingpong 把 softmax 藏在另一个 warpgroup 的 GEMM 后面（B4）。
--- step B200 {hl=b}
8192：比值 1.0。就算完美重叠，MUFU 也和 Tensor Core 一样忙。
--- step FA4 {hl=fa4}
把比例 f 的 exp 挪到 FMA（每 SM 128 次/时钟，一次模拟约 6 条指令）：两边相等时 f = 128 / (128 + 6·16) = 0.57（`balanced_emulation_fraction(16, 128, 6)`）。3 次多项式的最大相对误差 8.8e−5（在 [−30, 10] 上测得），远小于 BF16 的 2⁻⁹ ≈ 2e−3。真实 kernel 的 FMA 还要做别的事，所以 FA4 实际挪过去的比例更小。
:::

条件 rescale 的效果（8 个 query、512 个 key、块 32，由参考解算出）：threshold = 0、1、2、4、8 时，逐行 rescale 次数分别是 17、11、7、3、0，输出完全一致。

## 易错点

- **用截断代替 floor。** `int(−0.5) = 0`，小数部分变成 −0.5，多项式在 [0, 1) 外误差很大（`test_ex2_negative_fraction_uses_floor`）。
- **指数位加法溢出。** x = −1000 时 j << 23 会把符号位也改掉；先 `max(x, −127)`。
- **在 float64 里算多项式。** 题面要求全程 float32，结果 dtype 也要是 float32。
- **条件 rescale 时更新了 m 却没乘 acc。** “不 rescale” 意味着 m、acc、l 三个都不动；只改一部分结果就错了。
- **threshold = 0 时把 delta == 0 也算一次 rescale。** 最大值没变不需要 rescale，也不计数。

## 三级提示

::: hint 1 I/O 示例与关键问题
```python
ex2_emulation(np.float32(-3.3))     # 0.101535 （真值 0.1015315）
ex2_emulation(np.arange(-3, 4, dtype=np.float32))   # 精确的 2 的幂
O, n = softmax_attention_cond_rescale(Q, K, V, 0.5, 32, 8.0)    # O 精确，n = 0
exp_to_mma_ratio(8192, 16, 128)     # 1.0
```

#### 关键问题
1. 为什么 2^⌊x⌋ 可以用一次整数加法实现？
2. softmax 里的 x 几乎总是 ≤ 0，`fmax(x, −127)` 会不会影响结果？
3. 不 rescale 时 P 最大会到多少？为什么 FP32 不会溢出？
4. 为什么条件 rescale 不改变最终结果？
:::

::: hint 2 思路与常见坑
- `ex2_emulation`：`x = maximum(asarray(x, float32), -127)`；`j = floor(x)`；`f = x - j`；Horner：`((c3*f + c2)*f + c1)*f + c0`；`(p.view(int32) + (j.astype(int32) << 23)).view(float32)`。标量输入返回标量。
- 条件 rescale：log2 域 `s = scale·LOG2E·Q Kᵀ`；第一块直接设 m；之后 `delta = m - cand`，`keep = delta >= -threshold`（threshold > 0）或 `delta == 0`；`factor = where(keep, 1, exp2(delta))`；更新 acc、l、m；再加本块。
:::

::: hint 3 详解：答案
::: answer 关键问题答案
1. FP32 = 符号 · 2^(指数 − 127) · 1.尾数。p ∈ [1, 2) 的指数字段是 127；把 j 加到指数字段（位 23–30）上，就是乘 2^j。整数加法在 ALU / FMA 管线上做，不碰 MUFU。
2. 2⁻¹²⁷ 已经小于 FP32 最小正规数，再小对 softmax 的和没有任何影响（被 BF16 的 P 舍成 0）。
3. 新分数比 m 最多高 threshold（log2 域），所以 P ≤ 2^threshold = 256（threshold = 8）。l 是 P 的和，序列长 128K 也只到 2^25 量级，FP32 足够。
4. O = Σ 2^(s_j − m) v_j / Σ 2^(s_j − m)：分子分母同一个 m，2^(−m) 约掉。m 只要保证不溢出就行，不必是真正的最大值。
:::
:::

## 参考解

::: solution solutions/d3_fa4_softmax_tricks.py
{{code: solutions/d3_fa4_softmax_tricks.py}}
:::

## 面试官追问 & 对照源码

- **“为什么 FA4 不全部用多项式？”** FMA 管线也要做缩放、取最大值、类型转换；全部挪过去 FMA 就成了瓶颈。最优是让两条管线一样忙（`balanced_emulation_fraction`），实际由 `ex2_emu_freq` 按 fragment 交错。
- **“rescale_threshold 为什么是 8？”** log2 域的 8 = 256 倍：BF16 的 P 最大 256 不会损失精度（指数范围很大），FP32 的 l 和 O 也安全。更大的阈值收益很小。
- **“FA4 的 O 在 TMEM 里，rescale 怎么做？”** 专门的 correction warpgroup 用 `tcgen05.ld` 读 O、乘因子、`tcgen05.st` 写回；它和 MMA、softmax warp 通过 mbarrier 握手。条件 rescale 让它大多数时候什么都不用做。
- **“这两个技巧能用在 5090 上吗？”** 能：5090 的 Tensor Core 不快，exp 本来不是瓶颈（A1 的比值约 0.06 量级），收益很小；但条件 rescale 能省寄存器里的乘法，作为 G3 的扩展练习是个好起点。
