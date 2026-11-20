"""C4 · Between kernels: stream launches vs CUDA Graph vs Programmatic Dependent Launch (PDL) — your implementation (run ./check.sh c4).

一步 decode 是几百个 5–30 µs 的小 kernel。kernel 之间的空隙有三个来源：CPU 发射（每次几 µs）、
GPU 端的启动延迟（上一个 kernel 结束到下一个 kernel 的 CTA 开始跑）、以及下一个 kernel 自己的
prologue（分配 SMEM、初始化 mbarrier、预取 TMA descriptor、读权重）。
- CUDA Graph 一次性提交整条链：消掉 CPU 发射，并把 GPU 端空隙压小。
- PDL（sm90+）让下一个 kernel 在上一个 kernel 调用 `griddepcontrol.launch_dependents` 之后就启动，
  先跑不依赖上游输出的 prologue，到 `griddepcontrol.wait` 再等上游**全部完成**（含内存可见）。
所以 PDL 只有在 CPU 不是瓶颈时才有意义——也就是在 CUDA Graph 里（或 CPU 远远跑在前面时）。
这道题在假时钟上把这几种方式的时间线算出来，并检查 PDL 下哪些访存是不安全的。

给定（given）：Kernel(name, prologue, main, trigger=None)、Costs(cpu_launch, launch_latency, graph_launch, graph_gap)。
trigger 是 launch_dependents 在 main 阶段里的位置（0..1 的比例），None 表示从不显式触发（只在结束时隐式触发）。

timeline(kernels, mode, costs) -> dict：{"kernels": [{"start", "main_start", "end"}...], "total": 最后一个 end}。
记 trig(i) = main_start_i + trigger_i·main_i（trigger 为 None 时 = end_i）。每个 kernel：
main_start = start + prologue（PDL 模式见下），end = main_start + main。
- "stream"：CPU 依次发射，kernel i 在 cpu_i = (i+1)·cpu_launch 时发射完成；
  start_i = max(cpu_i, end_{i−1}) + launch_latency（i = 0 时没有 end_{−1} 项）。
- "stream_pdl"：同上，但 start_i = max(cpu_i, trig(i−1)) + launch_latency；
  main_start_i = max(start_i + prologue_i, end_{i−1})。
- "graph"：start_0 = graph_launch + launch_latency；start_i = end_{i−1} + graph_gap。
- "graph_pdl"：start_0 同上；start_i = trig(i−1) + graph_gap；main_start_i = max(start_i + prologue_i, end_{i−1})。
- 其他 mode 抛 ValueError。

pdl_hazards(ops, prev_writes, prev_reads) -> list[tuple[int, str, str]]：ops 是一个 PDL kernel 按程序顺序的操作，
元素为 "wait"、"trigger"、"compute"、"read:<buf>"、"write:<buf>"。在第一个 "wait" 之前：
读 prev_writes 里的 buf 是 "RAW"；写 prev_reads 里的 buf 是 "WAR"；写 prev_writes 里的 buf 是 "WAW"
（同时满足时报 "WAW"）。返回 (下标, op, 类型) 列表；wait 之后的操作都安全。

guaranteed_complete(i, waits) -> set[int]：kernel 0..n−1 以 PDL 链式启动，waits[j] 表示 kernel j 是否执行了
griddepcontrol.wait（kernel 0 是普通启动，视为 waits[0] = True，它之前没有 kernel）。
kernel i（i ≥ 1）通过它自己的 wait 之后，能确定已经完成的 kernel 集合：
G(i) = {i−1} ∪ (G(i−1) 若 waits[i−1] 为真，否则 ∅)，G(0) = ∅。

API
    Kernel, Costs                                         given
    timeline(kernels, mode, costs) -> dict
    pdl_hazards(ops, prev_writes, prev_reads) -> list[tuple[int, str, str]]
    guaranteed_complete(i, waits) -> set[int]
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Kernel:
    """一个 kernel：prologue（不读上游输出）、main（读上游输出），以及 launch_dependents 的位置。"""
    name: str
    prologue: float
    main: float
    trigger: float | None = None


@dataclass(frozen=True)
class Costs:
    """单位 µs。数值只是量级示意：CPU 每次 launch 几 µs，GPU 端启动延迟几 µs，graph 内空隙 ~1 µs。"""
    cpu_launch: float = 4.0
    launch_latency: float = 3.0
    graph_launch: float = 6.0
    graph_gap: float = 1.0




def timeline(kernels, mode: str, costs: Costs) -> dict:
    """几种发射方式下每个 kernel 的开始 / main 开始 / 结束时刻。"""
    raise NotImplementedError("timeline")


def pdl_hazards(ops, prev_writes, prev_reads) -> list[tuple[int, str, str]]:
    """griddepcontrol.wait 之前的不安全访存。"""
    raise NotImplementedError("pdl_hazards")


def guaranteed_complete(i: int, waits) -> set[int]:
    """kernel i 过了自己的 wait 之后，哪些更早的 kernel 一定已完成。"""
    raise NotImplementedError("guaranteed_complete")
