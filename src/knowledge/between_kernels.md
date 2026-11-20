lede: PDL、CUDA Graph、persistent kernel 都不改变 GEMM 的内循环，改的是 kernel 怎么被发射、怎么衔接。它们只有在“看得见整张图”的层面（CUDA graph 的录制者、编译器、推理框架）才能用好——这一页讲清楚为什么。
# Kernel 之间：CUDA Graph、PDL、persistent 与编译器

## 问题在哪

一步 decode 往往是几百个 5–30 µs 的小 kernel（每层的 RMSNorm、QKV GEMM、RoPE、attention、O-proj、MLP…）。每个 kernel 边界都有固定成本：

| 来源 | 量级 | 谁能消掉 |
|---|---|---|
| CPU 发射（Python + 驱动） | 几 µs / kernel | CUDA Graph |
| GPU 端启动延迟、上一个 kernel 的尾巴（部分 SM 已空闲） | 1–3 µs | PDL、persistent |
| 下一个 kernel 的 prologue（SMEM 划分、mbarrier 初始化、TMA 描述符预取、读权重） | 1–数 µs | PDL |
| 一次 HBM 往返（上一个写、下一个读） | 取决于大小 | 融合（epilogue fusion、torch.compile） |

10–20 µs 的 decode kernel，每个边界省几 µs 就是 10–20% 的端到端速度。

## 解决办法：三层工具

::: flow 从 CPU 到 SM：每一层消掉一段空隙
cfg cw=210 rh=100 nw=172
n cpu 0 0 "CPU 发射" "每个 kernel 几 µs" k=dark
n graph 1 0 "CUDA Graph" "整条链一次提交" k=acc
n pdl 2 0 "PDL" "prologue 藏进上游尾巴" k=pass
n fuse 3 0 "融合" "少一个 kernel" k=vio
n pers 1.5 1 "persistent / CLC" "CTA 常驻，循环取 tile" k=teal
e cpu -> graph
e graph -> pdl
e pdl -> fuse
e graph:b -> pers:l
cap: C4 在纸上算这几种时间线，G4 在 5090 上测：64 个小 kernel，stream 266 µs → stream+PDL 234 → graph 144 → graph+PDL 44。
:::

::: callout analogy 机场的登机口
每个 kernel 是一班飞机。**stream**：每班飞机起飞前塔台（CPU）都要单独批准一次。**CUDA Graph**：一整天的航班表一次批好。**PDL**：下一班的乘客在上一班还没起飞时就开始登机、放行李（prologue），只是舱门（griddepcontrol.wait）要等前一班真的推出了才关。**persistent**：飞机不回机库，落地加油接着飞下一段。**融合**：干脆把两段航程合成一段直飞。
:::

## PDL 的两半

- **device 端**：上游在合适的位置执行 `griddepcontrol.launch_dependents`（CUTLASS：[[cutlass:include/cutlass/arch/grid_dependency_control.h@launch_dependent_grids|launch_dependent_grids]]，GEMM 里放在 epilogue 附近）；下游在第一次读上游输出（或写上游输入）之前执行 `griddepcontrol.wait`（[[cutlass:include/cutlass/arch/grid_dependency_control.h@wait_on_dependent_grids|wait_on_dependent_grids]]）。
- **host 端**：`cudaLaunchKernelEx` 加 `cudaLaunchAttributeProgrammaticStreamSerialization`（[[cutlass:include/cutlass/kernel_launch.h#L105-L112|kernel_launch.h]]）。不加这个属性，device 端的指令是无害的空操作。
- **在 graph 里**：stream capture 会把这条依赖记成 programmatic edge，replay 时同样生效。

## 为什么说 PDL 是 graph / 编译器层面的优化

1. **需要三方配合**：上游要触发、下游要以 PDL 属性启动并把 wait 放对位置、中间不能有 CPU 同步或别的 stream 插队。单个 kernel 的作者只控制其中一方；能同时保证三者的，是录制 graph 的框架（SGLang 的 [[sglang:python/sglang/srt/model_executor/cuda_graph_runner.py@CudaGraphRunner|CudaGraphRunner]]）、生成 kernel 的编译器、或者一个库里成对设计的 kernel（[[sglang:sgl-kernel/python/sgl_kernel/utils.py@is_arch_support_pdl|sgl-kernel 默认开 PDL]]）。
2. **CPU 不能是瓶颈**：stream 模式下 kernel 比 CPU 发射还短时，下一个 kernel 根本还没被发出来，交接区用不上（C4 `test_pdl_cannot_fix_a_cpu_bound_stream`、G4 的 stream+PDL 只省 12%）。graph 把 CPU 去掉之后，剩下的正好是 PDL 能藏的那部分。
3. **正确性是全局性质**：`griddepcontrol.wait` 只等直接上游；读两层以前的 buffer 是否安全，取决于中间的 kernel 有没有 wait（C4 的 `guaranteed_complete`）。只有看得见整条链的人能判断。

## 编译器能替你做多少

- **融合**：CUTLASS 的 epilogue visitor tree 把 bias、激活、量化、残差加法融进 GEMM 的 epilogue；Triton / torch.compile（Inductor）把逐元素算子融进前后的 kernel。少一个 kernel = 少一次 HBM 往返 + 一次发射。
- **发射**：`torch.compile(mode="reduce-overhead")` 自动用 CUDA graph；PDL 要求生成的 kernel 自带 griddepcontrol，这部分正在逐步进入编译器。
- **调度**：persistent kernel + Stream-K（C3）由库在 kernel 内部完成；Blackwell 的 Cluster Launch Control 让 persistent 调度能适应被别的 kernel 占用的 SM。

## 和练习题的对应关系

| 题目 | 复刻了什么 | 源码 |
|---|---|---|
| C3 | wave quantization、persistent、Stream-K | [[cutlass:include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp|sm90 Stream-K]] |
| C4 | stream / graph / PDL 时间线、PDL 的依赖规则 | [[cutlass:include/cutlass/arch/grid_dependency_control.h|grid_dependency_control.h]] |
| G4 | 5090 上真实测量 PDL + CUDA Graph | [[sglang:sgl-kernel/csrc/elementwise/pos_enc.cuh#L103-L109|sgl-kernel 的 RoPE（griddepcontrol.wait）]] |
