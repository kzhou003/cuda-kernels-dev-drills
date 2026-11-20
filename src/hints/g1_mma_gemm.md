lede: 在 RTX 5090 上写一个真正跑在 Tensor Core 上的 BF16 GEMM：global → SMEM → ldmatrix → mma.sync → 寄存器累加 → 写回。A2 里在 numpy 上推过的 fragment 布局，这里每一个下标都要变成一行 CUDA。
# G1 · 5090：ldmatrix + mma.sync 的 BF16 GEMM

## 题面速览

- **在系统中的位置**：这是 Ampere / Ada / GeForce Blackwell 上所有 Tensor Core kernel 的骨架：CUTLASS sm80 GEMM、FA2、FlashInfer 的 prefill kernel。CuTe 里对应 [[cutlass:include/cute/arch/mma_sm80.hpp@SM80_16x8x16_F32BF16BF16F32_TN|SM80_16x8x16_F32BF16BF16F32_TN]] 和 [[cutlass:include/cute/arch/copy_sm75.hpp@SM75_U32x4_LDSM_N|SM75_U32x4_LDSM_N]]。sm120 的 MMA 仍是 warp 级 `mma.sync`（没有 wgmma / tcgen05），所以这套写法在 5090 上就是“正确答案”。
- **你要实现**：`mine/g1_mma_gemm.cu` 里 `// TODO(drill)` 处的主循环：每个 K 块把 A、B 搬进 SMEM，然后对两个 k16 步各做 4 次 A 的 `ldmatrix_x4`、2 次 B 的 `ldmatrix_x4`、16 次 `mma_16816`。`ldmatrix_x4`、`mma_16816`、`load_tile`、epilogue、host 函数都已给出。写完删掉 `#define DRILL_STUB` 那一行。
- **环境**：默认 RTX 5090（sm_120），任何 sm_80+ 都能跑。需要 torch（CUDA 版）+ nvcc：`PYTHON=/path/to/python-with-torch ./check.sh g1`。第一次编译 30–60 秒。
- **难点在这几条规则**：
  1. CTA tile 128×128×32，8 个 warp 排成 2（M）× 4（N），每个 warp 负责 64×32 = 4 个 m16 × 4 个 n8。
  2. A 的 ldmatrix 地址：lane l 给出第 `(l%8) + 8·((l/8)%2)` 行、第 `8·(l/16)` 列（A2 的 `a_tile_row_addrs`）。
  3. B 按 N×K 存放，一次 x4 读两个 n8 tile：行 `(l%8) + 8·(l/16)`、列 `8·((l/8)%2)`；寄存器 {0,1} 给第一个 n8，{2,3} 给第二个。
  4. 每个 K 块前后各一次 `__syncthreads()`：前一次保证数据到齐，后一次保证没人还在读就被下一块覆盖。

**验收要点**：6 种形状（含 1×1、尾块、2048³）对 FP32 参考 · 单位矩阵精确选行 · 结构化数值（C[i,n] = i）抓布局错误 · K 不是 32 的倍数时报错。

## 直觉与类比

::: callout analogy 一个车间、八个工位
CTA 是一个车间，SMEM 是车间中央的料架，8 个 warp 是 8 个工位，每个工位负责成品板（C）上 64×32 的一块。每一轮：所有人一起把一批原料（A 的 128×32、B 的 128×32）搬上料架（`load_tile`），等大家都搬完（`__syncthreads`），各工位按自己的位置从料架上取料：`ldmatrix` 是一个“智能取料臂”，你告诉它 8 行在哪，它直接把料按机床要的姿势塞进你的夹具（寄存器）；`mma.sync` 是机床，一下加工一块 16×8。下一批原料上架之前，要等所有人都取完（第二次 `__syncthreads`）。
:::

## 图解：一个 warp 在一个 k16 步里做的事

::: flow warp (warp_m, warp_n) 的 64×32 子块
cfg cw=200 rh=100 nw=168
n sa 0 0 "sA 128×40" "padding 到 40" k=teal
n sb 0 1 "sB 128×40" "N×K 存放" k=teal
n fa 1 0 "4 × ldmatrix_x4" "a[mi][4]" k=acc
n fb 1 1 "2 × ldmatrix_x4" "b[ni][2]" k=acc
n mma 2 0.5 "16 × mma_16816" "4 m16 × 4 n8" k=warn
n acc 3 0.5 "acc[4][4][4]" "64 个 FP32" k=pass
e sa -> fa
e sb -> fb
e fa -> mma
e fb -> mma
e mma -> acc
cap: SMEM 行跨度 40 个 bf16（80 B）：8 行同一列块的槽号是 `(5r + c) % 8`，8 个都不同，ldmatrix 无冲突（A3 的 padding 方案）。
:::

## 逐步演示

::: steps 一个 K 块（BK = 32）
::: flow
cfg cw=180 rh=90 nw=150
n ld 0 0 "load_tile × 2" "每线程 4 × 16 B" k=acc
n s1 1 0 "__syncthreads" "数据可见" k=warn
n k0 2 0 "kk = 0" "ldmatrix + 16 mma" k=pass
n k1 3 0 "kk = 16" "ldmatrix + 16 mma" k=pass
n s2 1.5 1 "__syncthreads" "允许覆盖" k=warn
e ld -> s1
e s1 -> k0
e k0 -> k1
e k1:b -> s2:r
e s2:l -> ld:b
:::
--- step 搬运 {hl=ld}
256 个线程，每个线程 `load_tile` 两次（A、B 各一次），每次 2 个 16 B 向量：128 行 × 4 个向量 = 512 个向量。行号 ≥ M（或 N）的写 0。
--- step 第一次同步 {hl=ld,s1}
普通的 global load + st.shared 没有异步完成机制，靠 `__syncthreads` 保证所有线程写完。
--- step k16 步 {hl=k0,k1}
每个 warp：A 4 条 `ldmatrix_x4`（4 个 m16），B 2 条（每条 2 个 n8），然后 4×4 = 16 条 `mma_16816`。每个 K 块两步，整个 CTA 每块共 8 warp × 32 = 256 条 mma。
--- step 第二次同步 {hl=s2}
下一轮的 `load_tile` 会覆盖 sA / sB；必须等所有 warp 读完。
:::

5090 上的速度（参考解，`./check.sh g1 --bench --solutions`，C 为 FP32；cuBLAS 是 `torch.matmul` 的 BF16）：

| M = N = K | 本题 TF/s | cuBLAS TF/s | 比值 |
|---|---|---|---|
| 1024 | 31.3 | 97.6 | 0.32 |
| 2048 | 136.2 | 155.3 | 0.88 |
| 4096 | 162.0 | 202.3 | 0.80 |
| 8192 | 182.9 | 210.7 | 0.87 |

1024 那一行慢是 C3 的 wave quantization：128×128 tile 只有 64 个 CTA，170 个 SM 只用了 38%。

## 易错点

- **A 的地址用了 `l / 8` 直接当行块。** 矩阵 j 的顺序是左上、左下、右上、右下：行偏移看 `(l/8) % 2`，列偏移看 `l / 16`。`test_structured_values_catch_layout_bugs` 用 C[i, n] = i 专门抓这类错误。
- **B 的两个 n8 搞反。** x4 的矩阵顺序是 (n 0–7, k 0–7)、(n 0–7, k 8–15)、(n 8–15, k 0–7)、(n 8–15, k 8–15)，所以行偏移看 `l / 16`、列偏移看 `(l/8) % 2`——和 A 正好相反。
- **少了第二次 `__syncthreads`。** 小矩阵上可能侥幸通过，大矩阵（2048³）上随机错误。
- **SMEM 行跨度用 32。** 结果对，但每条 ldmatrix 8 路冲突，慢一大截（A3）。
- **在 kernel 里用 `at::cuda::getCurrentCUDAStream`。** 部分 conda CUDA 工具链缺 cuSPARSE 头文件；本仓库统一用 `c10::cuda::getCurrentCUDAStream()`。

## 三级提示

::: hint 1 I/O 示例与关键问题
```python
A = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16)
B = torch.randn(384, 64, device="cuda", dtype=torch.bfloat16)
C = gemm_mma(A, B)                # (256, 384) float32 ≈ A.float() @ B.float().T
gemm_mma(A, torch.eye(64, device="cuda", dtype=torch.bfloat16)).equal(A.float())   # True
```

#### 关键问题
1. 一个 warp 的 64×32 输出需要几条 mma？每条多少 FLOP？
2. warp (warp_m, warp_n) 在 sA、sB 里从哪一行开始读？
3. 为什么 B 用 N×K（“TN”）存放就可以不转置地 ldmatrix？
4. 这个 kernel 在等什么？（提示：看两次 `__syncthreads` 之间的时间线）
:::

::: hint 2 思路与常见坑
```cuda
for (int k0 = 0; k0 < K; k0 += BK) {
  load_tile(sA, A + m0*K + k0, M - m0, K, tid);
  load_tile(sB, B + n0*K + k0, N - n0, K, tid);
  __syncthreads();
  for (int kk = 0; kk < BK; kk += 16) {
    // a[mi]: r = warp_m*64 + mi*16 + (lane%8) + 8*((lane/8)%2); c = kk + 8*(lane/16)
    // b: r = warp_n*32 + nj*16 + (lane%8) + 8*(lane/16); c = kk + 8*((lane/8)%2)
    // 16 x mma_16816(acc[mi][ni], a[mi], b[ni])
  }
  __syncthreads();
}
```
:::

::: hint 3 详解：答案
::: answer 关键问题答案
1. 每个 k16 步 4 × 4 = 16 条，每条 16·8·16·2 = 4096 FLOP。
2. sA 从第 `warp_m·64` 行、sB 从第 `warp_n·32` 行开始。
3. mma 的 B 操作数要“同一列 n 的相邻两个 k”在一个寄存器里。N×K、k 连续存放时，一个 8×8 子矩阵的一行就是一个 n 的 8 个连续 k，非转置 ldmatrix 正好把 (k=2t, 2t+1) 给 lane (g = n, t)。
4. 在等 global load：`load_tile` 是同步的，发出之后线程只能干等延迟（几百个周期），这段时间 Tensor Core 闲着。5090 上靠每个 SM 驻留 2 个 CTA 互相掩盖一部分；G2 用 cp.async 多级流水正面解决。
:::
:::

## 参考解

::: solution solutions/g1_mma_gemm.cu
{{code: solutions/g1_mma_gemm.cu}}
:::

Python 包装（`solutions/g1_mma_gemm.py`，mine/ 里一模一样）：

{{code: solutions/g1_mma_gemm.py}}

## 面试官追问 & 对照源码

- **“怎么把它变成 cuBLAS 的速度？”** 多级流水（G2）、更大的 warp tile（寄存器允许的话 64×64）、swizzle 代替 padding、寄存器双缓冲（下一个 k16 的 ldmatrix 和当前的 mma 交错）、tile 调度（C3）。
- **“FP16 累加会更快吗？”** 5090 上 FP16 累加的 Tensor Core 吞吐是 FP32 累加的两倍（GeForce 的限制），但精度差，LLM 一般不用。
- **“同一份代码在 H100 上跑多快？”** 远低于峰值：H100 需要每 SM 每时钟 32 FLOP/SMEM 字节（A1），warp 级 64×32 的 tile 只有 21；而且 mma.sync 没法异步。这正是 Hopper 要 wgmma + TMA 的原因。
- **“epilogue 能更快吗？”** 现在是每线程直接写两个 float，没有合并成 128 B 的事务。正规做法是先写 SMEM 再整行写回（或用 TMA store），A3 的 bank 冲突问题在这里也会出现。
