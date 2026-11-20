lede: 一步 decode 是几百个 5–30 µs 的小 kernel。kernel 写得再快，kernel 之间的空隙（CPU 发射、GPU 启动延迟、下一个 kernel 的 prologue）也会吃掉 10–30% 的时间。CUDA Graph 去掉 CPU，PDL 把 prologue 藏进上一个 kernel 的尾巴——但 PDL 只有在 graph 里才真正显出作用。
# C4 · Kernel 之间：stream vs CUDA Graph vs PDL 的时间线与依赖

## 题面速览

- **在系统中的位置**：device 端 [[cutlass:include/cutlass/arch/grid_dependency_control.h@launch_dependent_grids|launch_dependent_grids]] / [[cutlass:include/cutlass/arch/grid_dependency_control.h@wait_on_dependent_grids|wait_on_dependent_grids]]（`griddepcontrol.launch_dependents` / `griddepcontrol.wait`）；CUTLASS GEMM 只在 producer 第一次读 global 之前 wait（[[cutlass:include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp#L626-L629|pingpong.hpp L626–629]]）；host 端给 `cudaLaunchKernelEx` 加 `programmaticStreamSerializationAllowed`（[[cutlass:include/cutlass/kernel_launch.h#L105-L112|kernel_launch.h]]）。推理框架：SGLang 的 [[sglang:sgl-kernel/csrc/elementwise/pos_enc.cuh#L103-L109|RoPE kernel]] 开头先算索引再 `griddepcontrol.wait`，Python 侧默认 [[sglang:sgl-kernel/python/sgl_kernel/utils.py@is_arch_support_pdl|is_arch_support_pdl]]；整步 decode 用 [[sglang:python/sglang/srt/model_executor/cuda_graph_runner.py@CudaGraphRunner|CudaGraphRunner]] 录成 graph。G4 在 5090 上真测这四种方式。
- **你要实现**：`timeline(kernels, mode, costs)`（四种模式）、`pdl_hazards`（wait 之前哪些访存不安全）、`guaranteed_complete`（过了 wait 之后哪些上游一定完成了）。
- **难点在这几条规则**：
  1. PDL 下 kernel i 在上游**触发**后就能启动，但 main 必须等上游**全部完成**：`main_start = max(start + prologue, end_{i−1})`。
  2. 没有显式触发（trigger = None）时，隐式触发在上游结束时，PDL 什么也省不了。
  3. stream 模式下 CPU 每 `cpu_launch` 才发一个 kernel；kernel 比这还短时，CPU 是瓶颈，PDL 帮不上。
  4. `griddepcontrol.wait` 只等**直接上游**；传递性要靠中间的 kernel 自己也 wait 过。

**验收要点**：stream / graph / graph+PDL 的精确时间线 · 无触发时 PDL 无收益 · CPU 瓶颈下 PDL 收益很小而 graph 里收益大 · PDL 下 main 从不早于上游结束但启动更早 · 非法模式与空链 · RAW / WAR / WAW 检查 · 传递完成性。

## 直觉与类比

::: callout analogy 接力赛
每个 kernel 是一棒。**stream**：每一棒都要等教练（CPU）喊“下一个上场”，教练喊一次要 4 秒——棒越短，教练越成瓶颈。**CUDA Graph**：整场比赛的出场顺序提前排好，教练只喊一次“开始”。**PDL**：下一棒不用站在起点干等，而是在交接区里先跑起来（prologue：系鞋带、热身、拿好自己的装备 = 初始化 barrier、预取描述符、读权重），只是**接棒那一下**（griddepcontrol.wait）必须等上一棒真的到了。如果上一棒根本不发信号让你进交接区（不调用 launch_dependents），那你只能在起点等。而如果教练喊人比跑一棒还慢，交接区再长也没用——这就是“PDL 只有在 graph 里才显出作用”。
:::

## 图解：同一条链的三种时间线

::: flow 8 个 kernel，prologue 2 µs + main 6 µs
cfg cw=220 rh=100 nw=176
n st 0 0 "stream" "CPU 4 µs/次 → 92 µs" k=todo
n gr 1 0 "CUDA Graph" "间隔 1 µs → 80 µs" k=warn
n pdl 2 0 "Graph + PDL" "prologue 藏起来 → 59 µs" k=pass
n w 2.5 1 "griddepcontrol.wait" "等上游全部完成" k=vio
n tr 1.5 1 "launch_dependents" "让下游提前上 SM" k=acc
e st -> gr
e gr -> pdl
e tr:t -> pdl:b
e w:t -> pdl:b
cap: 数字来自 `timeline`（Costs 默认值：cpu_launch 4、launch_latency 3、graph_launch 6、graph_gap 1，trigger = 0 即在 main 开始时触发）。G4 在 5090 上测到的是同一个形状：64 个小 kernel，stream 266 µs → stream+PDL 234 → graph 144 → graph+PDL 44。
:::

## 逐步演示

::: steps graph + PDL，前两个 kernel
::: flow
cfg cw=200 rh=96 nw=166
n k0 0 0 "k0" "start 9 · main 11–17" k=acc
n tr 1 0 "k0 触发" "t = 11" k=warn
n k1 2 0 "k1 启动" "t = 12，prologue 12–14" k=pass
n wt 3 0 "k1 wait" "等到 k0 结束 17" k=vio
n m1 3 1 "k1 main" "17–23" k=pass
e k0 -> tr
e tr -> k1
e k1 -> wt
e wt -> m1
:::
--- step k0 {hl=k0}
graph 启动 6 + 启动延迟 3 → k0 在 9 开始，prologue 到 11，main 11–17。
--- step 触发 {hl=k0,tr}
k0 的 trigger = 0：main 一开始（11）就执行 `launch_dependents`。注意触发**不发布任何结果**，只是允许下游上 SM。
--- step k1 提前启动 {hl=tr,k1}
k1 在 11 + graph_gap = 12 启动，prologue 12–14 和 k0 的 main 重叠。
--- step wait {hl=k1,wt}
k1 在 14 到达 `griddepcontrol.wait`，等 k0 全部完成（17）。
--- step k1 main {hl=wt,m1}
k1 main 17–23。每个 kernel 的有效间隔从 graph 模式的 1 + 2 + 6 = 9 µs 缩到 6 µs：8 个 kernel 共 59 µs（graph 80，stream 92）。
:::

## 易错点

- **PDL 下 main 不等上游结束。** 最危险的 bug：读到上游还没写完的数据。`test_main_never_starts_before_previous_end_with_pdl`。
- **没触发也算提前启动。** trigger = None 时触发点就是上游结束（`test_pdl_without_trigger_gains_nothing`）。
- **stream_pdl 忽略了 CPU。** 下游启动 = `max(cpu_i, trig_{i−1}) + launch_latency`，CPU 没发出来的 kernel 不可能提前上 SM。
- **wait 之前写了上游还在读的 buffer。** ping-pong 或 in-place 的 kernel 很容易这样（WAR）；`pdl_hazards` 要报出来。
- **以为 wait 等的是“所有更早的 kernel”。** 只等直接上游；中间某个 kernel 不 wait 时传递性就断了（`test_transitive_completion_requires_intermediate_wait`）。

## 三级提示

::: hint 1 I/O 示例与关键问题
```python
ks = [Kernel(f"k{i}", 2.0, 6.0, 0.0) for i in range(8)]
timeline(ks, "stream", Costs())["total"]      # 92
timeline(ks, "graph", Costs())["total"]       # 80
timeline(ks, "graph_pdl", Costs())["total"]   # 59
pdl_hazards(["read:w", "read:x", "write:resid", "wait", "read:x"], {"x"}, {"resid"})
# [(1, 'read:x', 'RAW'), (2, 'write:resid', 'WAR')]
guaranteed_complete(2, [True, False, True])   # {1}
```

#### 关键问题
1. PDL 下一个 kernel 的有效时长是多少（相对于 graph）？
2. 为什么 CUTLASS 把 `launch_dependent_grids` 放在 epilogue 开始附近，而不是 kernel 入口？
3. 什么样的 prologue 工作可以放在 wait 之前？
4. torch.compile 的 `mode="reduce-overhead"` 和 PDL 是什么关系？
:::

::: hint 2 思路与常见坑
- `timeline`：对每个 kernel 先算 `start`（四种模式各一个公式），再 `main_start = start + prologue`，PDL 模式再和 `prev_end` 取 max，`end = main_start + main`，最后记下 `trig`。
- 输出的每个字典只要 `start / main_start / end` 三个键（`trig` 用完删掉）。
- `pdl_hazards`：遇到 `"wait"` 就停；`op.partition(":")` 拆出种类和 buffer。
- `guaranteed_complete`：递归；kernel i−1 是 0 或者 `waits[i−1]` 为真时继续往前。
:::

::: hint 3 详解：答案
::: answer 关键问题答案
1. 在 prologue 能完全藏住、上游及时触发的情况下：graph_gap 和 prologue 都被藏住，有效时长 ≈ main（本例 6 µs vs 9 µs）。
2. 下游 CTA 一旦启动就会占 SM 资源（寄存器、SMEM）。入口就触发会让下游 CTA 和上游的 mainloop 抢资源；在 epilogue 附近触发，下游的 prologue 正好和上游的尾巴（最后一波、epilogue 写回）重叠。G4 为了把效果放大，在入口触发。
3. 不读上游输出、也不写上游会读的东西的工作：计算索引、初始化 SMEM / mbarrier、预取 TMA 描述符、读权重 / scale / 位置编码表（它们不是上游写的）。
4. `reduce-overhead` 自动用 CUDA graph 录制，消掉 CPU 发射；PDL 是 graph 里 kernel 之间的边（programmatic edge），需要每个 kernel 自己带 `griddepcontrol` 指令并以 PDL 属性启动——编译器（Triton、Inductor）和框架（SGLang 的 sgl-kernel、FlashInfer 的 `enable_pdl`）正在逐步把它加上。
:::
:::

## 参考解

::: solution solutions/c4_launch_timeline.py
{{code: solutions/c4_launch_timeline.py}}
:::

## 面试官追问 & 对照源码

- **“为什么 PDL 只有在 graph / compiler 层面才真正有用？”** 单个 kernel 的作者无法让它生效：需要**上游**调用 launch_dependents、**下游**以 PDL 属性启动并在正确位置 wait、**中间**不能有 CPU 同步或别的流插进来。只有掌握整张图的人（CUDA graph 的录制者、编译器、推理框架）能保证这三件事同时成立，而且 CPU 发射开销被 graph 去掉之后，PDL 省下的 prologue 时间才成为主要部分。
- **“PDL 和 stream 上普通的 kernel fusion 怎么选？”** 能融合就融合（少一次 HBM 往返）；不能融合的（比如 GEMM → attention）用 PDL 把边界压小。CUTLASS 的 epilogue visitor tree 是 GEMM 方向的融合。
- **“Rubin 有什么新东西？”** 数据驱动同步：TMA 按 16 B 报告上游写的数据是否有效，下游按 tile 粒度等，而不是等整个 grid（[[cutlass:examples/121_rubin_data_driven_sync_gemm/121_rubin_data_driven_sync_gemm.cu|例子 121]]）。
- **“CUDA graph 的代价？”** 形状要固定（decode 按 batch size 分桶录多张图，见 `CudaGraphRunner.capture`）、显存要按最大形状预留、图里不能有 CPU 同步。
