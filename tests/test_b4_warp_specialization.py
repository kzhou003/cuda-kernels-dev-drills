import pytest

from drill import load

m = load("b4_warp_specialization")


def test_pingpong_short_epilogue_keeps_tensor_core_busy():
    tl = m.pingpong_timeline(4, 10, 6)
    assert [x["wg"] for x in tl] == [0, 1, 0, 1]
    assert [x["ml_start"] for x in tl] == [0, 10, 20, 30]
    assert m.makespan(tl) == 46
    assert m.tc_utilization(tl) == pytest.approx(40 / 46)


def test_pingpong_long_epilogue_waits_for_own_wg():
    tl = m.pingpong_timeline(4, 10, 15)
    # tile 2 (WG0) must wait for WG0's epilogue of tile 0: 10 + 15 = 25
    assert [x["ml_start"] for x in tl] == [0, 10, 25, 35]
    assert m.makespan(tl) == 60


def test_cooperative_serializes_epilogue():
    tl = m.cooperative_timeline(4, 10, 6)
    assert [x["ml_start"] for x in tl] == [0, 13, 26, 39]
    assert all(x["wg"] == -1 for x in tl)
    assert m.makespan(tl) == 52


def test_pingpong_beats_cooperative_when_epilogue_matters():
    for EP in (4, 8, 12):
        pp = m.makespan(m.pingpong_timeline(16, 10, EP))
        co = m.makespan(m.cooperative_timeline(16, 10, EP))
        assert pp < co


def test_empty_timeline():
    assert m.makespan([]) == 0
    assert m.tc_utilization([]) == 0.0


def test_timeline_invariants_tensor_core_never_shared():
    for ML, EP in [(10, 3), (7, 20), (5, 5)]:
        tl = m.pingpong_timeline(9, ML, EP)
        for a, b in zip(tl, tl[1:]):
            assert b["ml_start"] >= a["ml_end"]
        for wg in (0, 1):
            mine = [x for x in tl if x["wg"] == wg]
            for a, b in zip(mine, mine[1:]):
                assert b["ml_start"] >= a["ep_end"]


def test_setmaxnreg_matches_cutlass_pingpong():
    assert m.max_consumer_regs(40, 2) == 232
    assert m.max_consumer_regs(24, 2) == 240
    assert m.max_consumer_regs(40, 1) == 256        # capped
    assert 128 * (40 + 2 * 232) <= 65536


def test_setmaxnreg_validates_operand():
    for bad in (20, 41, 264):
        with pytest.raises(ValueError):
            m.max_consumer_regs(bad, 2)
