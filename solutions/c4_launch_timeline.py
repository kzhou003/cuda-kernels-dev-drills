"""C4 · Between kernels: stream launches vs CUDA Graph vs Programmatic Dependent Launch (PDL) — reference solution.

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


def _trig(k: Kernel, main_start: float, end: float) -> float:
    return end if k.trigger is None else main_start + k.trigger * k.main


def timeline(kernels, mode: str, costs: Costs) -> dict:
    """几种发射方式下每个 kernel 的开始 / main 开始 / 结束时刻。"""
    if mode not in ("stream", "stream_pdl", "graph", "graph_pdl"):
        raise ValueError(f"unknown mode {mode}")
    out = []
    pdl = mode.endswith("_pdl")
    for i, k in enumerate(kernels):
        prev = out[-1] if out else None
        if mode.startswith("stream"):
            cpu = (i + 1) * costs.cpu_launch
            if prev is None:
                start = cpu + costs.launch_latency
            else:
                # 没有 PDL：上一个 kernel 结束才轮到我。有 PDL：上一个触发后我就能上 SM。
                # 但 CPU 每 cpu_launch 才发出一个——kernel 比 cpu_launch 短时，CPU 才是瓶颈，PDL 救不了。
                gate = prev["trig"] if pdl else prev["end"]
                start = max(cpu, gate) + costs.launch_latency
        else:
            if prev is None:
                start = costs.graph_launch + costs.launch_latency
            else:
                start = (prev["trig"] if pdl else prev["end"]) + costs.graph_gap
        main_start = start + k.prologue
        if pdl and prev is not None:
            # griddepcontrol.wait：等上游 grid 全部完成且写回可见。prologue 被藏进了上游的尾巴里。
            main_start = max(main_start, prev["end"])
        end = main_start + k.main
        out.append({"start": start, "main_start": main_start, "end": end, "trig": _trig(k, main_start, end)})
    for o in out:
        o.pop("trig")
    return {"kernels": out, "total": out[-1]["end"] if out else 0.0}


def pdl_hazards(ops, prev_writes, prev_reads) -> list[tuple[int, str, str]]:
    """griddepcontrol.wait 之前的不安全访存。"""
    # 上游 kernel 可能还在跑：它写的东西我不能读（RAW），它读的东西我不能改（WAR），
    # 它写的东西我也不能写（WAW）。最常见的真实 bug：把 “读输入 x” 放到 wait 前面当 prologue；
    # 或者 in-place kernel 在 wait 前就写了上游还要读的残差 buffer。
    out = []
    pw, pr = set(prev_writes), set(prev_reads)
    for i, op in enumerate(ops):
        if op == "wait":
            break
        kind, _, buf = op.partition(":")
        if kind == "read" and buf in pw:
            out.append((i, op, "RAW"))
        elif kind == "write" and buf in pw:
            out.append((i, op, "WAW"))
        elif kind == "write" and buf in pr:
            out.append((i, op, "WAR"))
    return out


def guaranteed_complete(i: int, waits) -> set[int]:
    """kernel i 过了自己的 wait 之后，哪些更早的 kernel 一定已完成。"""
    # griddepcontrol.wait 只等“直接上游”那一个 grid。传递性靠上游自己也 wait 过：
    # 如果 kernel i−1 根本没调用 wait（它不读 i−2 的输出），它可能比 i−2 先结束，
    # 那 kernel i 就不能假设 i−2 的输出已经写好——读两层以前的 buffer 时要小心。
    if i <= 0:
        return set()
    g = {i - 1}
    if i - 1 == 0 or waits[i - 1]:
        g |= guaranteed_complete(i - 1, waits)
    return g
