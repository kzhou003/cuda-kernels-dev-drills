lede: H100 的 GEMM 里，warp 不再“既搬又算”：一个 producer warpgroup 只发 TMA，两个 consumer warpgroup 只做 wgmma 和 epilogue。Pingpong 让两个 consumer 轮流用 Tensor Core，一个做 epilogue 的时候另一个正好在算；setmaxnreg 把 producer 用不着的寄存器挪给 consumer。
# B4 · Warp specialization：Pingpong vs Cooperative 与 setmaxnreg

## 题面速览

- **在系统中的位置**：CUTLASS 的 [[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp|sm90_gemm_tma_warpspecialized_pingpong.hpp]] 和 [[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp|…_cooperative.hpp]]。pingpong 的寄存器分配写在 [[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp@LoadRegisterRequirement|LoadRegisterRequirement / MmaRegisterRequirement]]（40 / 232，重压力时 24 / 240），两个 consumer 用 `math_wg_order_barrier` 轮流进入 mainloop。FA3 在 H100 上用的是同一个思路。
- **你要实现**：`pingpong_timeline`、`cooperative_timeline`、`makespan`、`tc_utilization`、`max_consumer_regs`。
- **难点在这几条规则**：
  1. Tensor Core **同一时刻只服务一个 warpgroup**；pingpong 下 tile i 的 mainloop 必须等 tile i−1 的 mainloop 结束。
  2. 同一个 WG 的下一个 tile 还要等它自己的 epilogue 结束（累加器寄存器要先写出去）。
  3. cooperative：两个 WG 合算一个 tile，epilogue 两人各做一半（EP/2），这段时间 Tensor Core 空着。
  4. setmaxnreg：每个 warpgroup 128 线程，`128·(load + n·x) ≤ 65536`，x 是 ≤ 256 的 8 的倍数。

**验收要点**：短 epilogue 时 Tensor Core 不停 · 长 epilogue 时等自己的 WG · cooperative 串行 · epilogue 有分量时 pingpong 更快 · 空时间线 · Tensor Core 从不被两个 WG 同时占用 · 40 → 232、24 → 240 · setmaxnreg 操作数校验。

## 直觉与类比

::: callout analogy 两个厨师共用一口炒锅
一口大火炒锅（Tensor Core），两个厨师（consumer warpgroup），一个打杂（producer，只负责备菜 = TMA）。**Cooperative**：两个厨师一起炒一大锅，炒完一起装盘（epilogue），装盘时锅空着。**Pingpong**：厨师 A 炒自己那锅，炒完立刻把锅让给厨师 B，自己去装盘；B 炒完再让回给 A——只要装盘比炒菜快，锅就一直在火上。打杂的不需要大案板，于是把自己的案板空间（寄存器）让给两个厨师：40 + 232 + 232 刚好塞满 512（每线程寄存器预算 65536 / 128）。
:::

## 图解：一个 Hopper GEMM CTA 的分工

::: flow 1 个 producer + 2 个 consumer warpgroup
cfg cw=220 rh=100 nw=176
n prod 0 0.5 "producer WG" "TMA，40 个寄存器" k=acc
n ring 1 0.5 "SMEM ring" "full / empty barrier" k=teal
n wg0 2 0 "consumer WG0" "wgmma，232 个寄存器" k=pass
n wg1 2 1 "consumer WG1" "wgmma，232 个寄存器" k=pass
n tc 3 0.5 "Tensor Core" "顺序 barrier 轮流用" k=warn
e prod -> ring
e ring -> wg0
e ring -> wg1
e wg0 -> tc
e wg1 -> tc
cap: B2 的 mbarrier 环把 producer 和 consumer 连起来；pingpong 另外用一个 “order barrier” 让 WG0、WG1 交替进入 mainloop（[[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp#L850-L878|pingpong.hpp L850–878]]）。
:::

## 逐步演示

::: steps 4 个 tile，ML = 10，EP = 6
::: flow
cfg cw=190 rh=90 nw=160
n t0 0 0 "tile 0 · WG0" "ML 0–10" k=pass
n t1 1 0 "tile 1 · WG1" "ML 10–20" k=acc
n t2 2 0 "tile 2 · WG0" "ML 20–30" k=pass
n t3 3 0 "tile 3 · WG1" "ML 30–40" k=acc
n co 1.5 1 "cooperative" "13 一个 tile → 52" k=todo
e t0 -> t1
e t1 -> t2
e t2 -> t3
:::
--- step tile 0 {hl=t0}
WG0 从 0 开始 mainloop，10 时结束，接着做 epilogue（10–16）。
--- step tile 1 {hl=t0,t1}
WG1 等的是“Tensor Core 空出来”：10 时开始，20 结束，epilogue 20–26。WG0 的 epilogue 正好落在 WG1 的 mainloop 里。
--- step tile 2 {hl=t1,t2}
WG0 的条件：Tensor Core 20 空出，自己的 epilogue 16 已结束 → 20 开始。Tensor Core 一刻没停。
--- step 结束 {hl=t3}
tile 3 在 30–40，最后一个 epilogue 到 46：`makespan` = 46，利用率 40/46 = 0.87。
--- step 对比 cooperative {hl=co}
每个 tile 10 + 6/2 = 13，4 个 = 52，利用率 0.77。如果 EP = 15（长于 ML），pingpong 的 tile 2 要等 WG0 的 epilogue 到 25：开始时刻变成 0、10、25、35，makespan 60。
:::

## 易错点

- **pingpong 里只看 Tensor Core 不看自己的 WG。** EP > ML 时会算出“WG0 还在写累加器就开始下一个 tile”（`test_pingpong_long_epilogue_waits_for_own_wg`）。
- **cooperative 的 epilogue 用了 EP 而不是 EP/2。** 两个 WG 各写一半。
- **以为 pingpong 永远更好。** 每个 WG 的 tile 只有 cooperative 的一半大，强度更低（A1）；K 很大、epilogue 很短时 cooperative 往往更快。CUTLASS 两种都提供，由 profiler 选。
- **setmaxnreg 的值不是 8 的倍数，或超过 256。** ptxas 会直接报错（`test_setmaxnreg_validates_operand`）。
- **忘了 producer 也占寄存器。** 128·(40 + 2·232) = 64512，再多 8 个就超了。

## 三级提示

::: hint 1 I/O 示例与关键问题
```python
tl = pingpong_timeline(4, 10, 6)
[x["ml_start"] for x in tl]     # [0, 10, 20, 30]
makespan(tl), tc_utilization(tl) # (46, 0.869...)
makespan(cooperative_timeline(4, 10, 6))   # 52
max_consumer_regs(40, 2)        # 232
```

#### 关键问题
1. pingpong 下 Tensor Core 什么时候会空闲？
2. 为什么 cooperative 的 tile 可以更大？
3. producer 为什么只要 24–40 个寄存器？
4. setmaxnreg 之后 consumer 的 232 个寄存器里，累加器占多少？
:::

::: hint 2 思路与常见坑
- pingpong：维护 `tc_free`（上一个 mainloop 结束）和 `wg_free[2]`（每个 WG 上一个 epilogue 结束）；`s = max(tc_free, wg_free[i % 2])`。
- cooperative：一个变量 `t`，每个 tile `t → t + ML → t + ML + EP/2`。
- `max_consumer_regs`：先校验 load；`x = (65536 // 128 - load) // n`；向下取 8 的倍数；再 `min(256, …)`。
:::

::: hint 3 详解：答案
::: answer 关键问题答案
1. 当一个 WG 的 epilogue 比另一个 WG 的 mainloop 还长（EP > ML），或者只剩最后一个 tile 的 epilogue 时。
2. 两个 WG 合算 128×N 的 tile，每个 WG 的累加器是 64×N；pingpong 每个 WG 独占一个 tile，同样寄存器预算下 tile 是 64×N——强度更低。
3. 它只发 TMA：算坐标、发一条指令、操作 barrier，几乎不需要数据寄存器。
4. 64×256 的 FP32 累加器分给 128 线程 = 128 个寄存器（A1 的 `accumulator_regs_per_thread`），剩下的给地址、softmax（FA3）、epilogue 的转换。
:::
:::

## 参考解

::: solution solutions/b4_warp_specialization.py
{{code: solutions/b4_warp_specialization.py}}
:::

## 面试官追问 & 对照源码

- **“FA3 的 pingpong 和 GEMM 的有什么不同？”** FA3 让两个 warpgroup 交替做 “GEMM（QKᵀ / PV）” 和 “softmax”，因为 softmax（MUFU 上的 exp）在 H100 上已经占 MMA 时间的一半（D3）；另外还有 warpgroup 内的重叠：本块的 softmax 和下一块的 QKᵀ 交错。
- **“Blackwell 上还需要 pingpong 吗？”** 累加器在 TMEM 里、可以双缓冲（C1），MMA warp 和 epilogue warp 天然重叠，不需要把 tile 切小。CUTLASS sm100 的角色见 [[cutlass:include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp@WarpCategory|WarpCategory]]（MMA / Sched / MainloopLoad / EpilogueLoad / Epilogue）。
- **“DeepGEMM 的 FP8 kernel 怎么用 consumer 的寄存器？”** 每 128 个 K 把 wgmma 的部分和取出来乘 scale 再加进 FP32 累加器（C2），所以 consumer 需要两份累加器：[[deepgemm:deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh|sm90_fp8_gemm_1d2d.cuh]]。
- **“5090 能做 warp specialization 吗？”** 能（mbarrier + TMA 在 sm120 都有），但没有 wgmma，consumer 还是 warp 级 mma.sync，没有 setmaxnreg 的必要性那么强。
