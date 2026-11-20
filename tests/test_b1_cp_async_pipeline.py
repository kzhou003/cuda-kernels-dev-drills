import pytest

from drill import load

m = load("b1_cp_async_pipeline")


def test_wait_group_leaves_newest_groups_in_flight():
    q = m.CpAsyncQueue()
    for t in range(3):
        q.cp_async(t)
        q.commit_group()
    assert q.wait_group(2) == {0}
    assert q.wait_group(1) == {0, 1}
    assert q.wait_group(0) == {0, 1, 2}
    assert q.resident == {0, 1, 2}


def test_uncommitted_copies_are_not_waited():
    q = m.CpAsyncQueue()
    q.cp_async(7)
    assert q.wait_group(0) == set()
    q.commit_group()
    assert q.wait_group(0) == {7}


def test_empty_groups_count_as_groups():
    q = m.CpAsyncQueue()
    q.cp_async(0)
    q.commit_group()
    q.commit_group()          # empty
    assert q.wait_group(1) == {0}


def test_negative_wait_rejected():
    with pytest.raises(ValueError):
        m.CpAsyncQueue().wait_group(-1)


def test_ops_three_stages_four_tiles_exact():
    assert m.multistage_ops(4, 3) == [
        ("load", 0), ("commit",), ("load", 1), ("commit",),
        ("wait", 1), ("load", 2), ("commit",), ("mma", 0),
        ("wait", 1), ("load", 3), ("commit",), ("mma", 1),
        ("wait", 1), ("commit",), ("mma", 2),
        ("wait", 1), ("commit",), ("mma", 3),
    ]


def test_ops_single_stage_is_synchronous():
    assert m.multistage_ops(2, 1) == [("load", 0), ("commit",), ("wait", 0), ("mma", 0),
                                      ("load", 1), ("commit",), ("wait", 0), ("mma", 1)]
    with pytest.raises(ValueError):
        m.multistage_ops(2, 0)


@pytest.mark.parametrize("T,S", [(1, 2), (2, 4), (5, 2), (8, 3), (8, 5), (3, 6), (16, 4)])
def test_every_mma_reads_a_resident_tile(T, S):
    q = m.CpAsyncQueue()
    loaded = []
    for op in m.multistage_ops(T, S):
        if op[0] == "load":
            q.cp_async(op[1]); loaded.append(op[1])
        elif op[0] == "commit":
            q.commit_group()
        elif op[0] == "wait":
            q.wait_group(op[1])
        else:
            assert op[1] in q.resident, f"mma {op[1]} before its tile landed"
    assert sorted(loaded) == list(range(T))


@pytest.mark.parametrize("T,S", [(8, 3), (6, 4)])
def test_loads_never_overwrite_a_slot_still_needed(T, S):
    live = {}
    for op in m.multistage_ops(T, S):
        if op[0] == "load":
            slot = op[1] % S
            assert slot not in live, f"slot {slot} overwritten while tile {live.get(slot)} unused"
            live[slot] = op[1]
        elif op[0] == "mma":
            assert live.pop(op[1] % S) == op[1]


def test_simulate_synchronous():
    r = m.simulate(8, 1, 100, 40)
    assert r["total"] == 8 * 140
    assert r["mma_start"][:3] == [100, 240, 380]


def test_simulate_double_buffer_is_latency_bound():
    r = m.simulate(8, 2, 100, 40)
    assert r["total"] == 840
    assert r["mma_start"] == [100, 200, 300, 400, 500, 600, 700, 800]


def test_simulate_four_stages_hides_latency():
    r = m.simulate(8, 4, 100, 40)
    assert r["total"] == 420
    assert r["mma_start"] == [100, 140, 180, 220, 260, 300, 340, 380]
    assert r["util"] == pytest.approx(320 / 420)


def test_more_stages_never_slower():
    prev = None
    for S in range(1, 8):
        t = m.simulate(32, S, 300, 50)["total"]
        if prev is not None:
            assert t <= prev
        prev = t
    assert m.simulate(32, 7, 300, 50)["total"] == 300 + 32 * 50
