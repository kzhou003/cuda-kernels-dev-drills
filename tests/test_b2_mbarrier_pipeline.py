import random

import pytest

from drill import load

m = load("b2_mbarrier_pipeline")


def test_phase_flips_after_all_arrivals():
    b = m.MBarrier(3)
    assert b.phase == 0 and not b.try_wait(0) and b.try_wait(1)
    b.arrive(); b.arrive()
    assert b.phase == 0
    b.arrive()
    assert b.phase == 1 and b.completed == 1
    assert b.try_wait(0) and not b.try_wait(1)


def test_barrier_resets_itself_for_next_phase():
    b = m.MBarrier(2)
    b.arrive(2)
    b.arrive()
    assert b.phase == 1
    b.arrive()
    assert b.phase == 0 and b.completed == 2


def test_tx_bytes_hold_the_phase_open():
    b = m.MBarrier(1)
    b.arrive_and_expect_tx(1024)
    assert b.phase == 0
    b.complete_tx(512)
    assert b.phase == 0
    b.complete_tx(512)
    assert b.phase == 1


def test_complete_tx_may_arrive_before_expect():
    b = m.MBarrier(1)
    b.complete_tx(256)
    assert b.phase == 0
    b.arrive_and_expect_tx(256)
    assert b.phase == 1


def test_arrive_before_expect_would_complete_early():
    # this is why the producer must use arrive_and_expect_tx, not arrive then expect
    b = m.MBarrier(1)
    b.arrive()
    assert b.phase == 1


def test_pipeline_state_wraps_and_flips_phase():
    s = m.PipelineState(3)
    seen = []
    for _ in range(7):
        seen.append((s.index, s.phase))
        s.advance()
    assert seen == [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 0)]
    p = m.make_producer_start_state(4)
    assert (p.index, p.phase) == (0, 1)


def test_producer_first_lap_does_not_wait_but_second_lap_does():
    pipe = m.TmaPipeline(2, 1, 64)
    ps = m.make_producer_start_state(2)
    assert pipe.producer_try_acquire(ps); ps.advance()
    assert pipe.producer_try_acquire(ps); ps.advance()
    assert not pipe.producer_try_acquire(ps)          # slot 0 not released yet
    cs = m.PipelineState(2)
    assert not pipe.consumer_try_wait(cs)             # bytes not landed
    pipe.tma_complete(0, 64)
    assert pipe.consumer_try_wait(cs)
    pipe.consumer_release(cs)
    assert pipe.producer_try_acquire(ps)


def test_failed_acquire_changes_nothing():
    pipe = m.TmaPipeline(1, 1, 32)
    ps = m.make_producer_start_state(1)
    assert pipe.producer_try_acquire(ps); ps.advance()
    before = (pipe.full[0].phase, pipe.full[0].completed)
    assert not pipe.producer_try_acquire(ps)
    pipe.tma_complete(0, 32)
    assert pipe.full[0].completed == before[1] + 1


def _run(stages, consumers, n_tiles, seed):
    """Random interleaving of 1 producer, a TMA engine and `consumers` consumers."""
    rng = random.Random(seed)
    tx = 128
    pipe = m.TmaPipeline(stages, consumers, tx)
    slot_data = [None] * stages
    ps = m.make_producer_start_state(stages)
    produced = 0
    inflight = []                                  # (stage, tile, remaining chunks)
    cstate = [m.PipelineState(stages) for _ in range(consumers)]
    got = [[] for _ in range(consumers)]
    reading = [None] * consumers
    stall = 0
    while any(len(g) < n_tiles for g in got):
        actors = ["p", "t"] + [f"c{i}" for i in range(consumers)]
        a = rng.choice(actors)
        progressed = False
        if a == "p" and produced < n_tiles:
            if pipe.producer_try_acquire(ps):
                # the slot must not be in use by any consumer now
                for r in reading:
                    assert r is None or r[0] != ps.index, "producer acquired a slot still being read"
                inflight.append([ps.index, produced, 2])
                produced += 1
                ps.advance()
                progressed = True
        elif a == "t" and inflight:
            item = inflight[0]
            slot_data[item[0]] = item[1]           # bytes land in SMEM
            pipe.tma_complete(item[0], tx // 2)
            item[2] -= 1
            if item[2] == 0:
                inflight.pop(0)
            progressed = True
        elif a.startswith("c"):
            i = int(a[1:])
            s = cstate[i]
            if reading[i] is not None:
                idx, tile = reading[i]
                assert slot_data[idx] == tile, "slot overwritten while being read"
                got[i].append(tile)
                reading[i] = None
                pipe.consumer_release(s)
                s.advance()
                progressed = True
            elif len(got[i]) < n_tiles and pipe.consumer_try_wait(s):
                reading[i] = (s.index, slot_data[s.index])
                assert slot_data[s.index] == len(got[i]), "consumer saw stale data"
                progressed = True
        stall = 0 if progressed else stall + 1
        assert stall < 500, "deadlock"
    return got


@pytest.mark.parametrize("stages,consumers", [(1, 1), (2, 1), (3, 2), (4, 2), (5, 3)])
@pytest.mark.parametrize("seed", range(6))
def test_random_interleavings_deliver_every_tile_in_order(stages, consumers, seed):
    got = _run(stages, consumers, 25, seed)
    for g in got:
        assert g == list(range(25))
