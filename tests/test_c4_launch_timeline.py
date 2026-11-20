import pytest

from drill import load

m = load("c4_launch_timeline")
C = m.Costs()


def chain(n=8, pro=2.0, main=6.0, trig=0.0):
    return [m.Kernel(f"k{i}", pro, main, trig) for i in range(n)]


def test_stream_mode():
    r = m.timeline(chain(), "stream", C)
    assert r["kernels"][0] == {"start": 7, "main_start": 9, "end": 15}
    assert r["kernels"][1]["start"] == 18
    assert r["total"] == 92


def test_graph_mode():
    r = m.timeline(chain(), "graph", C)
    assert r["kernels"][0]["start"] == 9
    assert r["kernels"][1]["start"] == 18
    assert r["total"] == 80


def test_graph_pdl_hides_prologue_and_gap():
    r = m.timeline(chain(), "graph_pdl", C)
    k1 = r["kernels"][1]
    assert k1["start"] == 12                 # trig(k0) = 11, + gap
    assert k1["main_start"] == 17            # waits for k0 to end
    assert r["total"] == 59


def test_pdl_without_trigger_gains_nothing():
    a = m.timeline(chain(trig=None), "graph_pdl", C)["total"]
    b = m.timeline(chain(trig=None), "graph", C)["total"]
    assert a == b


def test_pdl_cannot_fix_a_cpu_bound_stream():
    # kernels shorter than the CPU launch cost: CPU is the bottleneck
    ks = chain(pro=0.5, main=1.0)
    s = m.timeline(ks, "stream", C)["total"]
    sp = m.timeline(ks, "stream_pdl", C)["total"]
    g = m.timeline(ks, "graph", C)["total"]
    gp = m.timeline(ks, "graph_pdl", C)["total"]
    assert gp < g < s
    # PDL saves far more inside a graph than in a CPU-bound stream
    assert (s - sp) / s < 0.1 and (g - gp) / g > 0.2


def test_main_never_starts_before_previous_end_with_pdl():
    for mode in ("stream_pdl", "graph_pdl"):
        r = m.timeline(chain(n=5, pro=4, main=3, trig=0.0), mode, C)
        ks = r["kernels"]
        for a, b in zip(ks, ks[1:]):
            assert b["main_start"] >= a["end"]
            if mode == "graph_pdl":
                assert b["start"] < a["end"]      # but it did launch early


def test_bad_mode_and_empty():
    with pytest.raises(ValueError):
        m.timeline(chain(), "magic", C)
    assert m.timeline([], "graph", C)["total"] == 0


def test_hazards_before_wait():
    ops = ["read:weights", "read:x", "write:resid", "compute", "wait", "read:x", "write:y"]
    hz = m.pdl_hazards(ops, prev_writes={"x"}, prev_reads={"resid", "w0"})
    assert hz == [(1, "read:x", "RAW"), (2, "write:resid", "WAR")]


def test_waw_and_no_wait_at_all():
    hz = m.pdl_hazards(["write:x", "trigger", "read:x"], prev_writes={"x"}, prev_reads={"x"})
    assert hz == [(0, "write:x", "WAW"), (2, "read:x", "RAW")]
    assert m.pdl_hazards(["wait", "read:x"], {"x"}, set()) == []


def test_transitive_completion_requires_intermediate_wait():
    assert m.guaranteed_complete(0, [True]) == set()
    assert m.guaranteed_complete(3, [True, True, True, True]) == {0, 1, 2}
    # kernel 1 did not wait: kernel 2 only knows kernel 1 is done
    assert m.guaranteed_complete(2, [True, False, True]) == {1}
    assert m.guaranteed_complete(3, [True, True, False, True]) == {2}
    assert m.guaranteed_complete(1, [True, False]) == {0}
