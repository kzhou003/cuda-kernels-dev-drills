"""B2 · mbarrier: phase parity, transaction bytes and the TMA producer/consumer ring — reference solution.

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
        # 硬件里就是 SMEM 里的一个 64-bit 字：pending 计数、tx 计数和相位位挤在一起。
        # 线程们等的不是“值”，而是“相位位翻了没有”——所以同一个 barrier 可以被反复使用，
        # 不需要任何人去重置它：完成的瞬间硬件自己把计数装回初值。
        self.expected = expected_arrivals
        self._pending = expected_arrivals
        self._tx = 0
        self.phase = 0
        self.completed = 0

    def _maybe_complete(self) -> None:
        if self._pending == 0 and self._tx == 0:
            self.phase ^= 1
            self.completed += 1
            self._pending = self.expected
            self._tx = 0

    def arrive(self, count: int = 1) -> None:
        """到达 count 次。"""
        self._pending -= count
        self._maybe_complete()

    def expect_tx(self, nbytes: int) -> None:
        """声明本相位还要等 nbytes 个事务字节。"""
        # expect_tx 本身不检查完成：pending 还没到 0 时完成不了；
        # 若 complete_tx 已经先到（tx 为负），这次 expect 把它拉回 0，下一次 arrive 就完成。
        self._tx += nbytes

    def arrive_and_expect_tx(self, nbytes: int) -> None:
        """producer 的标准动作：先登记字节，再到达。"""
        # 顺序很重要：如果先 arrive，pending 变 0 而 tx 还是 0，相位会在 TMA 一个字节都没写的时候就完成，
        # consumer 读到的是旧数据。PTX 的 mbarrier.arrive.expect_tx 把两步做成一条原子指令。
        self._tx += nbytes
        self.arrive(1)

    def complete_tx(self, nbytes: int) -> None:
        """TMA（或 bulk copy）写完 nbytes 时由硬件调用。"""
        self._tx -= nbytes
        self._maybe_complete()

    def try_wait(self, parity: int) -> bool:
        """parity 对应的相位是否已经完成。"""
        # 只有一位相位，所以只能区分“当前相位”和“上一个相位”。
        # 这就要求等待方永远不能落后两圈——ring buffer 的 empty/full 配对天然保证了这一点。
        return parity != self.phase


class PipelineState:
    def __init__(self, stages: int, index: int = 0, phase: int = 0):
        """ring buffer 上的位置：第几个 stage，以及这是第几圈（只记奇偶）。"""
        self.stages = stages
        self.index = index
        self.phase = phase

    def advance(self) -> None:
        """前进一格；绕回 0 时翻转 phase。"""
        # 每绕一圈，同一个 stage 的 barrier 恰好完成一个相位，所以我们等的 parity 也要翻转。
        self.index += 1
        if self.index == self.stages:
            self.index = 0
            self.phase ^= 1


def make_producer_start_state(stages: int) -> PipelineState:
    """producer 从 phase 1 开始：empty barrier 的“上一个相位”视为已完成，所以第一圈不用等。"""
    return PipelineState(stages, 0, 1)


class TmaPipeline:
    def __init__(self, stages: int, num_consumers: int, tx_bytes: int):
        """每个 stage 一对 barrier：full（1 次 arrive + tx 字节）和 empty（num_consumers 次 arrive）。"""
        self.stages = stages
        self.tx_bytes = tx_bytes
        self.full = [MBarrier(1) for _ in range(stages)]
        self.empty = [MBarrier(num_consumers) for _ in range(stages)]

    def producer_try_acquire(self, state: PipelineState) -> bool:
        """等 empty；拿到后对 full 登记 tx_bytes 并到达。"""
        if not self.empty[state.index].try_wait(state.phase):
            return False
        # 拿到 slot 之后、发 TMA 之前登记字节数：full 的完成条件变成“producer 到了 + 字节全到”。
        self.full[state.index].arrive_and_expect_tx(self.tx_bytes)
        return True

    def tma_complete(self, stage: int, nbytes: int) -> None:
        """TMA 引擎写完一段。"""
        self.full[stage].complete_tx(nbytes)

    def consumer_try_wait(self, state: PipelineState) -> bool:
        """数据到齐了吗？"""
        return self.full[state.index].try_wait(state.phase)

    def consumer_release(self, state: PipelineState) -> None:
        """这个 consumer 读完了这个 slot。"""
        # 所有 consumer 都 release 之后 empty 才翻相位，producer 才能覆盖。
        # Hopper 的 wgmma 是异步的：release 必须放在 warpgroup_wait 之后，
        # 否则 Tensor Core 可能还在读这块 SMEM（CUTLASS 的 mma 里 release 的是“上一块”）。
        self.empty[state.index].arrive()
