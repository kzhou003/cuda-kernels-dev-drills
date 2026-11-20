"""B2 · mbarrier: phase parity, transaction bytes and the TMA producer/consumer ring — your implementation (run ./check.sh b2).

Hopper 起，SMEM 里的 `mbarrier` 是 kernel 内部一切异步协作的“信号灯”：TMA 拷贝完成时按字节数
给它记账（complete_tx），MMA warp 等它翻相位才开始读；读完再 arrive 另一个 mbarrier 通知 producer
这块 SMEM 可以覆盖了。CUTLASS 的 `PipelineTmaAsync` 用每个 stage 一对 barrier（full / empty）
加一个 `PipelineState(index, phase)` 把这一切串成环形缓冲。这道题从零实现这三样东西。

MBarrier(expected_arrivals)：
- 内部状态：本相位还差几次 arrive（pending，初值 expected_arrivals）、还差多少事务字节（tx，初值 0）、
  当前相位位 phase（初值 0）。
- arrive(count=1)：pending −= count。
- expect_tx(nbytes)：tx += nbytes。
- arrive_and_expect_tx(nbytes)：先 expect_tx 再 arrive(1)。
- complete_tx(nbytes)：tx −= nbytes（允许暂时为负：TMA 的完成可能早于 expect 被看到）。
- 每次 arrive / complete_tx 之后，若 pending == 0 且 tx == 0，本相位完成：phase ^= 1，
  pending 重置为 expected_arrivals，tx 重置为 0，completed 计数 +1。
- try_wait(parity) -> bool：parity 指的那个相位是否已经完成，等价于 `parity != phase`
  （PTX `mbarrier.try_wait.parity` 的语义：初始 phase=0 时等 parity 1 立即返回真）。
- 属性 phase、completed（已完成的相位数）。

PipelineState(stages, index=0, phase=0)：advance() 使 index += 1，到 stages 时回到 0 并翻转 phase。
make_producer_start_state(stages)：返回 index 0、phase 1 的状态（缓冲一开始就是空的，producer 不必等）。

TmaPipeline(stages, num_consumers, tx_bytes)：每个 stage 一个 full = MBarrier(1)、
一个 empty = MBarrier(num_consumers)。
- producer_try_acquire(state) -> bool：empty[state.index].try_wait(state.phase) 为真时，
  对 full[state.index] 调用 arrive_and_expect_tx(tx_bytes) 并返回 True；否则什么都不做，返回 False。
- tma_complete(stage, nbytes)：模拟 TMA 引擎写完 nbytes：full[stage].complete_tx(nbytes)。
- consumer_try_wait(state) -> bool：full[state.index].try_wait(state.phase)。
- consumer_release(state)：empty[state.index].arrive()。
- 属性 full、empty（MBarrier 列表）。

API
    MBarrier(expected_arrivals)  .arrive(count=1) .expect_tx(n) .arrive_and_expect_tx(n) .complete_tx(n)
                                 .try_wait(parity) -> bool  .phase  .completed
    PipelineState(stages, index=0, phase=0)  .advance()  .index  .phase
    make_producer_start_state(stages) -> PipelineState
    TmaPipeline(stages, num_consumers, tx_bytes)  .producer_try_acquire(state) -> bool
                 .tma_complete(stage, n)  .consumer_try_wait(state) -> bool  .consumer_release(state)
"""
from __future__ import annotations


class MBarrier:
    def __init__(self, expected_arrivals: int):
        """expected_arrivals 次 arrive + 事务字节清零 = 一个相位完成。"""
        raise NotImplementedError("MBarrier.__init__")


    def arrive(self, count: int = 1) -> None:
        """到达 count 次。"""
        raise NotImplementedError("MBarrier.arrive")

    def expect_tx(self, nbytes: int) -> None:
        """声明本相位还要等 nbytes 个事务字节。"""
        raise NotImplementedError("MBarrier.expect_tx")

    def arrive_and_expect_tx(self, nbytes: int) -> None:
        """producer 的标准动作：先登记字节，再到达。"""
        raise NotImplementedError("MBarrier.arrive_and_expect_tx")

    def complete_tx(self, nbytes: int) -> None:
        """TMA（或 bulk copy）写完 nbytes 时由硬件调用。"""
        raise NotImplementedError("MBarrier.complete_tx")

    def try_wait(self, parity: int) -> bool:
        """parity 对应的相位是否已经完成。"""
        raise NotImplementedError("MBarrier.try_wait")


class PipelineState:
    def __init__(self, stages: int, index: int = 0, phase: int = 0):
        """ring buffer 上的位置：第几个 stage，以及这是第几圈（只记奇偶）。"""
        raise NotImplementedError("PipelineState.__init__")

    def advance(self) -> None:
        """前进一格；绕回 0 时翻转 phase。"""
        raise NotImplementedError("PipelineState.advance")


def make_producer_start_state(stages: int) -> PipelineState:
    """producer 从 phase 1 开始：empty barrier 的“上一个相位”视为已完成，所以第一圈不用等。"""
    raise NotImplementedError("make_producer_start_state")


class TmaPipeline:
    def __init__(self, stages: int, num_consumers: int, tx_bytes: int):
        """每个 stage 一对 barrier：full（1 次 arrive + tx 字节）和 empty（num_consumers 次 arrive）。"""
        raise NotImplementedError("TmaPipeline.__init__")

    def producer_try_acquire(self, state: PipelineState) -> bool:
        """等 empty；拿到后对 full 登记 tx_bytes 并到达。"""
        raise NotImplementedError("TmaPipeline.producer_try_acquire")

    def tma_complete(self, stage: int, nbytes: int) -> None:
        """TMA 引擎写完一段。"""
        raise NotImplementedError("TmaPipeline.tma_complete")

    def consumer_try_wait(self, state: PipelineState) -> bool:
        """数据到齐了吗？"""
        raise NotImplementedError("TmaPipeline.consumer_try_wait")

    def consumer_release(self, state: PipelineState) -> None:
        """这个 consumer 读完了这个 slot。"""
        raise NotImplementedError("TmaPipeline.consumer_release")
