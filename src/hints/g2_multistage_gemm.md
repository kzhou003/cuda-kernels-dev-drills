lede: 把 G1 的同步搬运换成 cp.async 多级 ring buffer：prologue 先发 Stages−1 块，之后每轮“等最老的一块 → 发最新的一块 → 算”。在 5090 上你还会测到一个反直觉的结果：2 级比 4 级快——因为 SMEM 多了，occupancy 就少了。
# G2 · 5090：cp.async 多级流水 GEMM（2 / 3 / 4 级）

## 题面速览

- **在系统中的位置**：CUTLASS sm80 的 [[cutlass:include/cutlass/gemm/collective/sm80_mma_multistage.hpp|sm80_mma_multistage.hpp]]（[[cutlass:include/cutlass/gemm/collective/sm80_mma_multistage.hpp#L299-L335|主循环 L299–335]]：`cp_async_wait<Stages-2>()` → `__syncthreads()` → 发下一块 → `cp_async_fence()` → 算），PTX 是 [[cutlass:include/cute/arch/copy_sm80.hpp@SM80_CP_ASYNC_CACHEGLOBAL_ZFILL|SM80_CP_ASYNC_CACHEGLOBAL_ZFILL]]。FA2 的 K/V 加载用的也是这套。B1 是它的 CPU 版。
- **你要实现**：`mine/g2_multistage_gemm.cu` 里的 prologue 和主循环（`template <int STAGES>`，2 / 3 / 4 都要对）。`cp_async_16`（带 zero-fill）、`cp_async_commit`、`cp_async_wait<N>`、`issue_tile`、`warp_mma_k16`（就是 G1 的 k16 步）、epilogue、host 端（含 `cudaFuncSetAttribute` 打开 > 48 KB 的动态 SMEM）都已给出。
- **环境**：默认 RTX 5090（sm_120）；任何 sm_80+、每 CTA ≥ 80 KB SMEM 的 GPU 都行（A100 / H100 / L40 都可以）。
- **难点在这几条规则**：
  1. prologue：对 s = 0..S−2，s < KT 才发，但**每次都 commit**。
  2. 主循环每轮：`cp_async_wait<S-2>` → `__syncthreads` → 若 `kt+S−1 < KT` 发到 slot `(kt+S−1) % S` → **总是 commit** → 算 slot `kt % S`。
  3. 尾行用 src-size = 0 的 zero-fill，地址仍要合法（钳到第 0 行）。
  4. 退出前 `cp_async_wait<0>()`：不能有拷贝还在飞时 CTA 就结束。

**验收要点**：6 种形状 × 3 种 stage 数 · 单位矩阵与结构化数值 · K 块数少于 / 多于 stage 数（KT = 1..128）× 3 种 stage · 每个 K 块落进正确的 slot（C = 32·Σ块号）· 非法 K、非法 stage 数报错。

## 直觉与类比

::: callout analogy 传送带有多长，桌子就要多大
B1 的回转寿司：传送带上放 S−1 盘就能藏住厨房的延迟。但传送带占地方：每多一格就多 20 KB 的 SMEM。5090 每个 SM 只有 100 KB 的“店面”：40 KB 的 2 级传送带能开两家店（2 个 CTA），60 或 80 KB 的只能开一家。一家店里传送带再长，也比不上两家店交替营业——当厨房（Tensor Core）本来就不快的时候。H100 上情况相反：店面 228 KB、灶台快 8 倍，就需要更长的传送带。
:::

## 图解：stage 数、SMEM 和 occupancy

::: flow 5090：每 SM 100 KB SMEM、65536 个寄存器
cfg cw=210 rh=100 nw=172
n s2 0 0 "2 级 · 40 KB" "每 SM 2 个 CTA" k=pass
n s3 1 0 "3 级 · 60 KB" "每 SM 1 个 CTA" k=warn
n s4 2 0 "4 级 · 80 KB" "每 SM 1 个 CTA" k=warn
n reg 0 1 "寄存器" "256 线程 × ~128" k=dark
n res 2 1 "结果" "2 级最快 ≈ 0.9× cuBLAS" k=teal
e s2 -> s3
e s3 -> s4
e reg -> s2
e s4 -> res
cap: `DRILL_OCC=1` 时 host 端会打印 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 的结果：2 级 2 个 CTA/SM，3、4 级 1 个。A1 的 `ctas_per_sm` 给出同样的数。
:::

## 逐步演示

::: steps STAGES = 3，KT = 4（K = 128）
::: flow
cfg cw=180 rh=90 nw=150
n p 0 0 "prologue" "发 0、1" k=acc
n w 1 0 "wait<1> + sync" "块 kt 到齐" k=warn
n i 2 0 "发 kt+2" "slot (kt+2)%3" k=acc
n c 3 0 "算 slot kt%3" "2 个 k16 步" k=pass
e p -> w
e w -> i
e i -> c
e c:b -> w:b
:::
--- step prologue {hl=p}
`issue_tile` 块 0 → slot 0、块 1 → slot 1，各 commit 一次。2 组在飞。
--- step kt = 0 {hl=w,i,c}
`wait<1>`：最老的一组（块 0）到齐；`__syncthreads` 让所有线程的拷贝可见。发块 2 → slot 2，commit。算 slot 0。
--- step kt = 1 {hl=w,i,c}
`wait<1>` → 块 1 到齐。发块 3 → slot 0（块 0 已经被所有 warp 读完，因为大家都过了这一轮的 `__syncthreads`）。算 slot 1。
--- step kt = 2、3 {hl=w,c}
没有新块：**空 commit**。`wait<1>` 依次保证块 2、3 到齐。少了空 commit，kt = 3 时 `wait<1>` 只保证到块 2。
--- step 收尾 {hl=c}
`cp_async_wait<0>()`，然后 epilogue（和 G1 一样）。
:::

5090 上的速度（参考解，`./check.sh g2 --bench --solutions`）：

| M = N = K | 2 级 TF/s | 3 级 | 4 级 | cuBLAS | 2 级 / cuBLAS |
|---|---|---|---|---|---|
| 2048 | 140.6 | 126.0 | 126.1 | 154.9 | 0.91 |
| 4096 | 178.3 | 153.9 | 154.7 | 201.5 | 0.89 |
| 8192 | 189.1 | 168.2 | 169.5 | 209.0 | 0.90 |

（对照 G1：同步版本在 8192 上是 182.9。2 级 cp.async 只快 3%，因为 5090 上 2 个 CTA 交替已经藏住了大部分延迟。）

## 易错点

- **尾部不 commit 空组。** KT < STAGES−1 或最后几轮会读到没到齐的块；`test_k_tiles_fewer_and_more_than_stages` 用 K = 32..160 专门测。
- **wait 之后没有 `__syncthreads`。** `wait_group` 只管本线程发的拷贝；别的线程搬的那一半看不到。
- **新块写进 `kt % STAGES`。** 那是本轮要算的块：`test_each_k_tile_lands_in_the_right_slot` 会得到错误的和。
- **predicated-off 的拷贝用了越界地址。** 即使 src-size = 0，地址也会被检查；钳到合法行。
- **忘了 `cudaFuncSetAttribute`。** 超过 48 KB 的动态 SMEM 启动直接失败（已在 host 端给出，但自己写 kernel 时常忘）。

## 三级提示

::: hint 1 I/O 示例与关键问题
```python
C = gemm_pipelined(A, B, stages=2)    # 2、3、4 都应该和 A.float() @ B.float().T 一致
gemm_pipelined(A, B, 5)               # RuntimeError: stages must be 2, 3 or 4
```
```bash
DRILL_OCC=1 ./check.sh g2 --bench     # 打印每种 stage 的 CTA/SM 和 TF/s
```

#### 关键问题
1. 第 kt 轮 wait 之前一共 commit 了几组？
2. 为什么一次 `__syncthreads` 同时解决了“可见性”和“slot 复用”两个问题？
3. 为什么 cp.async 能用 `.cg`（只走 L2）？
4. 什么时候多级流水在 5090 上会明显赢？
:::

::: hint 2 思路与常见坑
```cuda
const int KT = K / BK;
for (int s = 0; s < STAGES - 1; ++s) {
  if (s < KT) { issue_tile(sA + s*TILE_ELEMS, gA + s*BK, ...); issue_tile(sB + ...); }
  cp_async_commit();
}
for (int kt = 0; kt < KT; ++kt) {
  cp_async_wait<STAGES - 2>();
  __syncthreads();
  int nxt = kt + STAGES - 1;
  if (nxt < KT) { int slot = nxt % STAGES; issue_tile(...); issue_tile(...); }
  cp_async_commit();
  int cur = kt % STAGES;
  for (int kk = 0; kk < BK; kk += 16) warp_mma_k16(sA + cur*TILE_ELEMS, sB + cur*TILE_ELEMS, kk, ...);
}
cp_async_wait<0>();
```
:::

::: hint 3 详解：答案
::: answer 关键问题答案
1. (STAGES−1) + kt 组；`wait<STAGES−2>` 后块 0..kt 到齐（B1）。
2. 所有线程都到达 barrier 时：(a) 每个线程都已经 wait 过自己的拷贝，barrier 让它们彼此可见；(b) 每个 warp 都已经算完上一轮（读完了 slot `(kt−1) % S`），所以这一轮可以覆盖它。
3. A、B 的每个元素只被这个 CTA 搬进 SMEM 一次，之后都从 SMEM 读；放进 L1 没有复用价值，还会挤掉别的数据。
4. 当每 SM 只能放 1 个 CTA 时（比如更大的 tile 128×256、或寄存器用得更多），或者 K 很大、访存延迟更长（数据不在 L2 里）时。H100 / A100 上 3–5 级是标配。
:::
:::

## 参考解

::: solution solutions/g2_multistage_gemm.cu
{{code: solutions/g2_multistage_gemm.cu}}
:::

Python 包装（`solutions/g2_multistage_gemm.py`）：

{{code: solutions/g2_multistage_gemm.py}}

## 面试官追问 & 对照源码

- **“怎么让 3 级也放下 2 个 CTA？”** 把 padding 换成 swizzle（每 stage 省 20%：48 KB）、或者 BK 从 32 改成 16。这是一个很好的扩展练习：改完用 `DRILL_OCC=1` 验证。
- **“寄存器级的双缓冲是什么？”** 在一个 K 块内，先 ldmatrix 第 kk+16 步的 fragment，再算第 kk 步的 mma，让 ldmatrix 的延迟藏在 mma 后面（CUTLASS 的 `K_BLOCK_MAX` 循环）。
- **“在 5090 上怎么用 TMA？”** sm120 有 TMA（目标 `shared::cta`）和 mbarrier：把 `issue_tile` 换成一个线程发 `cp.async.bulk.tensor`，用 B2 的 full / empty barrier 代替 wait_group + `__syncthreads`。描述符要在 host 端用 `cuTensorMapEncodeTiled` 建，SMEM 用 128B swizzle（B3）。
- **“为什么 cuBLAS 在 8192 上能到 210？”** 更大的 tile、寄存器双缓冲、swizzle、persistent 调度，以及针对 sm120 调过的参数；剩下的 10% 大多在这里。
