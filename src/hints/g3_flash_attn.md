lede: 把 D1 / D2 的递推写进 Tensor Core 的寄存器 fragment：S = QKᵀ 用 mma.sync，行最大值和行和在 quad 里用两次 shuffle 归约，P 不经过 SMEM 直接变成下一次 mma 的 A 操作数，V 用转置的 ldmatrix 读。这就是 FlashAttention-2 在 Ampere / 5090 上的核心。
# G3 · 5090：FlashAttention-2 前向（mma.sync + online softmax）

## 题面速览

- **在系统中的位置**：FA2 的 [[fa:csrc/flash_attn/src/flash_fwd_kernel.h@compute_attn_1rowblock|compute_attn_1rowblock]]（CuTe 写法）；softmax 部分是 [[fa:csrc/flash_attn/src/softmax.h@softmax_rescale_o|softmax_rescale_o]]，quad 内归约是 [[fa:csrc/flash_attn/src/softmax.h@quad_allreduce_|quad_allreduce_]]，P 的寄存器重排是 [[fa:csrc/flash_attn/src/utils.h@convert_layout_acc_Aregs|convert_layout_acc_Aregs]]。FA 的 CuTe DSL 版本也有 sm120 的前向：[[fa:flash_attn/cute/flash_fwd.py|flash_fwd.py]]（sm80 路径）。
- **你要实现**：`mine/g3_flash_attn.cu` 里的 KV 循环：S = QKᵀ、缩放到 log2 域并 mask、online softmax（含 O 的 rescale）、P·V，以及最后 l 的 quad 归约。Q fragment 的加载、ldmatrix（含 `.trans`）、mma、`pack_bf16`、`load_rows`、写回 O 和 LSE、host 端都已给出。
- **环境**：默认 RTX 5090；任何 sm_80+。head_dim 64 或 128，Q / K / V 形状 (B, H, N, D) 相同，BF16。
- **难点在这几条规则**：
  1. 一个 CTA = 一个 (batch·head, 64 个 query)，4 个 warp 各 16 行；每个 warp 看全部 key（FA2 的分法，不需要 warp 间归约）。
  2. 每个线程持有第 g、g+8 两行；一行的 64 个 S 分布在一个 quad 的 4 个线程上 → `__shfl_xor_sync` 1 和 2。
  3. P 的 C fragment 直接打包成 A fragment：k16 步 kk 用 n8 tile `2kk` 和 `2kk+1`（A2 的“行对齐”）。
  4. V 是 [key][d]（K×N、n 连续）：B 操作数要用 `ldmatrix.trans`。
  5. causal：KV 块只扫到对角线；只有对角线块逐元素 mask；尾部 key（≥ N）总是 mask。

**验收要点**：D = 64 / 128 × N = 64 / 128 / 200 / 1000 × causal / 非 causal 的 O 和 LSE · 长度 1、7、63、65 · causal 第一行 = V 的第一行 · 放大 8 倍的分数不溢出 · 自定义 scale · 不支持的 head_dim 报错。

## 直觉与类比

::: callout analogy 4 个记账员，每人管 16 个客户
一个 CTA 是一个小账房，4 个记账员（warp）每人管 16 个客户（query 行）。每一批新到的交易记录（64 个 key 的 K、V）放在公共桌上（SMEM）。记账员先用计算器（mma）算出自己 16 个客户对这 64 笔交易的打分（S），然后按 D1 的“换汇率”规则更新每个客户的账：同一行的数据分在 4 个助手（quad 的 4 个线程）手里，助手之间喊两嗓子（两次 shuffle）就知道这一行的最大值。最妙的一步：打分表（S 的 C 布局）和下一次计算要的输入格式（A 布局）恰好对得上，直接把 float 压成 bf16 就能拿去算 P·V，一张纸都不用往桌上放。
:::

## 图解：一个 warp 处理一个 KV 块

::: flow 16 行 query × 64 个 key
cfg cw=200 rh=100 nw=168
n q 0 0 "qf[D/16][4]" "Q fragment，常驻" k=acc
n k 0 1 "sK [key][d]" "ldmatrix（非转置）" k=teal
n s 1 0.5 "s[8][4]" "S，C 布局" k=warn
n sm 2 0.5 "online softmax" "quad shuffle × 2" k=vio
n p 3 0 "pa[4]" "bf16，A 布局" k=pass
n v 2 1.4 "sV [key][d]" "ldmatrix.trans" k=teal
n o 3 1.4 "o[D/8][4]" "O 累加器" k=pass
e q -> s
e k -> s
e s -> sm
e sm -> p
e p:b -> o:t
e v -> o
cap: 寄存器预算（D = 128）：qf 32 个、s 32 个、o 64 个、m / l 4 个，再加地址，每线程 ~150 个寄存器，128 线程的 CTA。
:::

## 逐步演示

::: steps 一个 KV 块（64 个 key），一个 warp，D = 64
::: flow
cfg cw=180 rh=90 nw=150
n mm1 0 0 "S = QKᵀ" "4 kd × 4 nj × 2 mma" k=acc
n msk 1 0 "× scale·log2e" "mask → −inf" k=warn
n soft 2 0 "max / α / exp2 / l" "每线程 2 行" k=vio
n mm2 3 0 "O += P V" "4 kk × 4 nd × 2 mma" k=pass
e mm1 -> msk
e msk -> soft
e soft -> mm2
:::
--- step S = QKᵀ {hl=mm1}
D = 64：4 个 k16 步（kd）× 4 次 `ldmatrix_x4`（每次 2 个 n8）× 2 次 mma = 32 条 mma，得到 `s[8][4]`：每个线程 8 个 n8 tile × 4 个元素。
--- step 缩放和 mask {hl=msk}
乘 `scale_log2 = scale · log2 e`，之后全部用 `exp2f`。元素 (ni, e) 的 key = `k0 + 8ni + 2t + (e&1)`，query = `q0 + 16warp + g + 8(e>>1)`。key ≥ N 或（对角线块上）key > query → −inf。
--- step online softmax {hl=soft}
对 h = 0、1（行 g、g+8）：本线程 16 个值取 max，`__shfl_xor_sync(…, 1)`、`(…, 2)` 得到整行 max；`α = exp2(m_old − m_new)`；`s = exp2(s − m_new)`；`l = α·l + Σs`（l 只是本线程的部分和，最后再 quad 归约）；O 的第 h 行乘 α。
--- step O += P V {hl=mm2}
k16 步 kk：`pa = {pack(s[2kk][0,1]), pack(s[2kk][2,3]), pack(s[2kk+1][0,1]), pack(s[2kk+1][2,3])}`；V 用 `ldmatrix_x4_trans`，地址行 = `16kk + (l%8) + 8((l/8)%2)`、列 = `16nd + 8(l/16)`，寄存器 {0,1} 给 n8 tile 2nd，{2,3} 给 2nd+1。
:::

5090 上的速度（参考解，B = 4、H = 16，对比 PyTorch SDPA）：

| N | D | causal | 本题 TF/s | SDPA TF/s | 比值 |
|---|---|---|---|---|---|
| 1024 | 64 | 否 | 139.7 | 162.2 | 0.86 |
| 4096 | 64 | 否 | 113.5 | 211.5 | 0.54 |
| 8192 | 128 | 否 | 121.8 | 208.0 | 0.59 |
| 8192 | 128 | 是 | 129.4 | 203.1 | 0.64 |

差距主要在 K/V 是同步加载的：等 global load 时 Tensor Core 闲着。把 G2 的 cp.async 双缓冲用到 K/V 上，是下一步最值钱的扩展。

## 易错点

- **只在一个线程里求行最大值。** 一行分布在 quad 的 4 个线程上，必须 shuffle 两次；少一次结果偶尔对（max 碰巧在本线程）大多不对。
- **l 在每块都做 quad 归约。** 正确但浪费；而且如果每块都归约，最后就不能再归约一次（会重复相加）。参考解只在最后归约一次。
- **第一块的 α。** m_old = −inf：`exp2(−inf − m)` = 0 正确；但如果整行都是 −inf（m_new 也是 −inf），要用 0 做基准，否则 NaN（`test_tiny_and_odd_lengths` 里 N = 1、7 的 padding 行会触发）。
- **V 用了非转置 ldmatrix。** 结果是 P·Vᵀ 的某种乱序；`test_matches_reference` 立刻失败。
- **causal 判断写成 `key >= query`。** 对角线自己要保留：`test_first_causal_row_copies_first_value` 检查第一行输出等于 V 的第一行。
- **每个 KV 块开头少了 `__syncthreads`。** 上一块的 sV 还有 warp 在读，就被下一块覆盖。

## 三级提示

::: hint 1 I/O 示例与关键问题
```python
q, k, v = (torch.randn(2, 3, 200, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
o, lse = flash_attn(q, k, v, causal=True)    # o: (2,3,200,64) bf16, lse: (2,3,200) fp32
# o ≈ softmax(q kᵀ / 8 + causal mask) v；lse ≈ logsumexp(q kᵀ / 8 + mask)
```

#### 关键问题
1. 线程 (warp, lane) 的 `s[ni][e]` 对应哪个 query、哪个 key？
2. 为什么 P 可以不经过 SMEM 直接做 A 操作数？V 为什么不行？
3. causal 时一个 Q 块要扫几个 KV 块？哪一块要 mask？
4. LSE 怎么从 log2 域的 m 和 l 算出来？
:::

::: hint 2 思路与常见坑
```cuda
const int n_blocks = causal ? min(ceil(N/BC), ceil((q0+BR)/BC)) : ceil(N/BC);
for (jb ...) {
  __syncthreads(); load_rows(sK ...); load_rows(sV ...); __syncthreads();
  float s[BC/8][4] = {};
  // S: for kd, for nj: ldmatrix_x4(sK + r*LD + kd*16 + 8*((lane/8)%2)), r = nj*16 + (lane%8) + 8*(lane/16)
  // scale + mask
  // for h in 0,1: row max (2 shuffles), base = (mx == -inf ? 0 : mx), alpha = exp2f(m[h] - base),
  //               s = exp2f(s - base), l[h] = l[h]*alpha + rowsum, o[*][2h, 2h+1] *= alpha, m[h] = mx
  // PV: pa from s, ldmatrix_x4_trans(sV + r*LD + c), 2 mma per nd
}
// l: two shuffles
```
:::

::: hint 3 详解：答案
::: answer 关键问题答案
1. query = `q0 + 16·warp + g + 8·(e >> 1)`，key = `k0 + 8·ni + 2t + (e & 1)`（A2 的 C 布局：c0、c1 在行 g，c2、c3 在行 g+8）。
2. P 是 S 经过逐元素运算得到的，仍在 C 布局；C 布局和 A 布局按行对齐（A2 的 `test_c_layout_rows_match_a_layout_rows`），只要两两打包成 bf16。V 来自 SMEM，而且要作为 B 操作数（k = key 方向成对），V 存成 [key][d]，所以用转置的 ldmatrix。
3. `ceil((q0 + 64) / 64)` 个，最后一个（对角线块）要逐元素 mask；N 有尾巴时最后一块还要 mask key ≥ N。
4. 自然对数的 lse = ln Σ e^{x} = (m₂ + log₂ l)·ln 2，m₂ 是 log2 域的最大值（已经乘过 scale·log2 e）。
:::
:::

## 参考解

::: solution solutions/g3_flash_attn.cu
{{code: solutions/g3_flash_attn.cu}}
:::

Python 包装（`solutions/g3_flash_attn.py`）：

{{code: solutions/g3_flash_attn.py}}

## 面试官追问 & 对照源码

- **“FA2 真实代码比这里多了什么？”** cp.async 双缓冲 K/V、swizzle 的 SMEM layout、按 `Is_even_MN` / `Is_even_K` 生成无判断的快路径、dropout、ALiBi、softcap、paged KV、varlen（cu_seqlens），以及更大的 Br（128）和 8 个 warp。
- **“反向怎么做？”** 存下 LSE，反向重算 P = exp(S − lse)，再算 dV = Pᵀ dO、dP = dO Vᵀ、dS = P ∘ (dP − rowsum(dO ∘ O))、dQ = dS K、dK = dSᵀ Q。dQ 要跨 KV 块累加，FA2 用 atomicAdd 或单独的 dQ 累加 kernel。
- **“5090 上 FP8 attention？”** sm120 有 FP8 的 `mma.sync`（k = 32），FA3 的 FP8 思路（Q、K 量化、P 用 FP8 做 PV）可以移植；SageAttention 是一个现成的例子。
- **“decode（N_q = 1）用这个 kernel 行吗？”** 64 行的 Q 块只有 1 行有效，浪费 63/64 的 Tensor Core；decode 要用 split-KV（D1）+ 按 head 并行的专门 kernel（FlashDecoding、FlashInfer 的 decode）。
