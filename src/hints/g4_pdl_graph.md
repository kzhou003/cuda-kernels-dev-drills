lede: 在 5090 上把 C4 的时间线测出来：一条 64 个小 kernel 的链，分别用普通 stream、stream + PDL、CUDA Graph、Graph + PDL 发射。你要写的只有几行——griddepcontrol 放在哪、cudaLaunchKernelEx 怎么配——但它们决定了 kernel 之间省下 3/4 的时间还是一点都省不下。
# G4 · 5090：PDL + CUDA Graph，测出 kernel 之间省下的时间

## 题面速览

- **在系统中的位置**：device 端就是 CUTLASS 的 [[cutlass:include/cutlass/arch/grid_dependency_control.h@launch_dependent_grids|launch_dependent_grids]] / [[cutlass:include/cutlass/arch/grid_dependency_control.h@wait_on_dependent_grids|wait_on_dependent_grids]]；host 端是 [[cutlass:include/cutlass/kernel_launch.h#L105-L112|kernel_launch.h]] 里的 `cudaLaunchAttributeProgrammaticStreamSerialization`。SGLang 的 RMSNorm / RoPE 等 kernel 都带 `enable_pdl`（[[sglang:sgl-kernel/csrc/elementwise/pos_enc.cuh#L103-L109|pos_enc.cuh]]），整步 decode 录成 CUDA graph（[[sglang:python/sglang/srt/model_executor/cuda_graph_runner.py@CudaGraphRunner|CudaGraphRunner]]）。
- **你要实现**：`mine/g4_pdl_graph.cu` 里两处 TODO：(1) device 端 `step_kernel` 里 `pdl_launch_dependents()`、prologue（读 w、`spin_cycles`）、`pdl_wait()` 的位置；(2) host 端用 `cudaLaunchKernelEx` + PDL 属性发射 n_kernels 个 kernel（ping-pong 两个 buffer）。
- **环境**：PDL 需要 **sm_90+**（H100、B200、RTX 50 系都行）；默认 RTX 5090。graph 测试用 `torch.cuda.graph`。
- **难点在这几条规则**：
  1. `launch_dependents` 只是“允许下游上 SM”，**不发布结果**；下游的 wait 仍然等整个上游 grid 完成。所以可以放在入口。
  2. `wait` 要放在第一次碰上游数据**之前**：读 x（上游的输出，RAW）和写 y（上游正在读的输入，WAR）都算。
  3. host 端没有 PDL 属性时，griddepcontrol 指令是无害的空操作；有属性时它们才生效。
  4. 在 stream capture 里发射同样有效：graph 里的边会变成 programmatic edge。

**验收要点**：PDL 开 / 关 × 1、2、7、64 个 kernel 的结果都等于闭式解，且返回正确的 buffer · 不开 PDL 时下一个 kernel 从不早于上一个结束 · 开 PDL 时 15 次交接里至少 12 次提前开始（用 `%globaltimer` 时间戳）· graph 里录 10 个 kernel、replay 3 次 = 30 步。

## 直觉与类比

::: callout analogy 接力赛的交接区（C4 的续）
C4 是在纸上画接力赛，这里是真的跑一场：64 棒，每棒在接棒前有一段“热身”（`spin_cycles(4000)` ≈ 1.7 µs，代表 barrier 初始化、描述符预取、读权重）。stream：教练（CPU）每喊一棒要几 µs。graph：出场顺序提前排好。PDL：下一棒在交接区里先热身，接棒那一刻（`griddepcontrol.wait`）才真正等。你会看到：没有 graph 时 PDL 只省 12%；有 graph 时 PDL 再省 70%。
:::

## 图解：一个 kernel 的三段

::: flow step_kernel（PDL 版）
cfg cw=210 rh=100 nw=172
n trig 0 0 "launch_dependents" "入口：下游可以上 SM" k=acc
n pro 1 0 "prologue" "读 w，spin" k=teal
n wait 2 0 "griddepcontrol.wait" "等上游 grid 完成" k=warn
n main 3 0 "main" "y = a·x + w" k=pass
n host 0 1 "cudaLaunchKernelEx" "PDL 属性 = 1" k=dark
e trig -> pro
e pro -> wait
e wait -> main
e host -> trig
cap: x 是上游写的（读它之前必须 wait），y 是上游读的（ping-pong，写它之前也必须 wait）。w 只有自己读，可以放在 prologue。
:::

## 逐步演示

::: steps 64 个 kernel，每个 prologue 4000 周期（RTX 5090 实测）
::: flow
cfg cw=190 rh=90 nw=160
n s 0 0 "stream" "266.5 µs" k=todo
n sp 1 0 "stream + PDL" "233.9 µs" k=warn
n g 2 0 "CUDA graph" "143.8 µs" k=acc
n gp 3 0 "graph + PDL" "43.6 µs" k=pass
e s -> sp
e sp -> g
e g -> gp
:::
--- step stream {hl=s}
每个 kernel 4.16 µs：CPU 发射 + GPU 启动延迟 + prologue 1.7 µs + 很短的 main，全部串行。
--- step stream + PDL {hl=s,sp}
3.66 µs / kernel，只省 12%：CPU 发射仍然一个一个来，下一个 kernel 常常在上一个结束后才被 CPU 发出来，交接区用不上。
--- step CUDA graph {hl=g}
2.25 µs / kernel：CPU 开销没了，剩下 GPU 端的启动间隔 + prologue。
--- step graph + PDL {hl=g,gp}
0.68 µs / kernel——比 prologue 本身（1.7 µs）还短：因为每个 kernel 一进来就触发，下游的下游也能接着提前上 SM，多个 kernel 的 prologue 叠在一起跑，只有 main 是串行的。这就是“PDL 只有在 graph 里才真正有用”的实测版本。
:::

（数字来自 `./check.sh g4 --bench --solutions`，你的机器上会有出入；`spin` 越大，PDL 的收益越明显。）

## 易错点

- **把读 x 放进 prologue。** 能跑、多数时候结果也对，偶尔错——典型的竞态。测试不一定每次都抓得到，所以要靠 C4 的 `pdl_hazards` 这种推理。
- **在 wait 之前写 y。** y 是上一个 kernel 正在读的 x（ping-pong），WAR 竞态。
- **忘了 host 端的属性。** 指令照样执行但没有效果：`test_with_pdl_next_kernel_starts_before_previous_ends` 会发现没有重叠。
- **`<<<>>>` 发射。** 三重尖括号语法没法带属性，必须用 `cudaLaunchKernelEx`。
- **stamps 指针没按 kernel 偏移。** 第 k 个 kernel 写 `stamps + 2k`。
- **以为触发越早越好。** 这里为了放大效果在入口触发；真实 GEMM 在 epilogue 附近触发，避免下游 CTA 和上游 mainloop 抢 SMEM / 寄存器。

## 三级提示

::: hint 1 I/O 示例与关键问题
```python
x0 = torch.randn(100_000, device="cuda"); w = torch.randn_like(x0)
b0, b1 = x0.clone(), torch.empty_like(x0)
out = run_chain(b0, b1, w, 7, 0.5, 2000, True)   # out is b1; out == 7 steps of x = 0.5*x + w
st = torch.zeros(32, dtype=torch.int64, device="cuda")
run_chain(b0, b1, w, 16, 0.5, 20000, True, st)   # st.view(-1, 2): [start_ns, end_ns] per kernel
```

#### 关键问题
1. `launch_dependents` 放在入口安全吗？为什么？
2. prologue 里能读 w 吗？能读 x 吗？能写 y 吗？
3. 为什么测试用 `%globaltimer` 而不是 `clock64()`？
4. 不开 PDL 时，为什么下一个 kernel 的开始时间一定晚于上一个 kernel block 0 的结束时间？
:::

::: hint 2 思路与常见坑
device：
```cuda
if (use_pdl) pdl_launch_dependents();
const float wi = i < n ? w[i] : 0.f;
spin_cycles(spin);
if (use_pdl) pdl_wait();
```
host：
```cuda
cudaLaunchConfig_t cfg = {};  cfg.gridDim = ...; cfg.blockDim = 256; cfg.stream = stream;
cudaLaunchAttribute attr[1];
attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
attr[0].val.programmaticStreamSerializationAllowed = use_pdl;
cfg.attrs = attr; cfg.numAttrs = 1;
for (k ...) cudaLaunchKernelEx(&cfg, step_kernel, x, y, w, n, a, spin, use_pdl, st ? st + 2*k : nullptr);
```
:::

::: hint 3 详解：答案
::: answer 关键问题答案
1. 安全：触发只影响“下游什么时候能被调度”，下游的 `griddepcontrol.wait` 依然要等本 grid 全部完成且内存可见。代价只是下游 CTA 更早占用 SM 资源。
2. w 能读（没有任何 kernel 写它）；x 不能（上游的输出，RAW）；y 不能写（上游的输入，WAR）。
3. `clock64()` 是每个 SM 自己的周期计数器，不同 SM、不同 kernel 之间不可比；`%globaltimer` 是全局纳秒计时器。
4. 没有 PDL 时，同一个 stream 上下一个 kernel 要等上一个 kernel 的所有 block 都结束才启动，block 0 的结束时间早于（或等于）整个 grid 的结束。
:::
:::

## 参考解

::: solution solutions/g4_pdl_graph.cu
{{code: solutions/g4_pdl_graph.cu}}
:::

Python 包装（`solutions/g4_pdl_graph.py`）：

{{code: solutions/g4_pdl_graph.py}}

## 面试官追问 & 对照源码

- **“为什么说 PDL 是 graph / compiler 层面的优化？”** 见 C4：要上下游三方配合、CPU 发射开销要先被 graph 去掉。本题的四个数字就是证据。
- **“torch.compile 会自动加 PDL 吗？”** `mode="reduce-overhead"` 自动用 CUDA graph；PDL 需要每个 kernel 自己有 griddepcontrol 并以属性启动，Triton 生成的 kernel 和 Inductor 正在逐步支持。推理框架（SGLang、FlashInfer、TensorRT-LLM）是手动在关键 kernel 上加的。
- **“PDL 和多 stream 并发有什么区别？”** 多 stream 是没有依赖的 kernel 并发；PDL 是**有**依赖的 kernel 部分重叠（依赖之前的部分）。
- **“上游 kernel 有 persistent 调度时呢？”** persistent kernel 的 CTA 一直占着 SM，下游提前启动也没地方放；所以 CUTLASS 在 persistent kernel 的最后一个 tile 才触发。
