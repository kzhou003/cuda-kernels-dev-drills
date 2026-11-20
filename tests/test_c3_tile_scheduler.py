import random
from collections import Counter

import pytest

from drill import load

m = load("c3_tile_scheduler")


def test_wave_quantization_h100():
    w = m.wave_stats(200, 132)
    assert w["waves"] == 2 and w["last_wave"] == 68
    assert w["efficiency"] == pytest.approx(200 / 264)


def test_wave_stats_exact_fit_and_multi_cta():
    assert m.wave_stats(264, 132) == {"waves": 2, "last_wave": 132, "efficiency": 1.0}
    assert m.wave_stats(200, 132, ctas_per_sm=2)["waves"] == 1
    assert m.wave_stats(0, 132) == {"waves": 0, "last_wave": 0, "efficiency": 1.0}


def test_raster_order_swizzle_1_is_row_major():
    assert m.raster_order(2, 3) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]


def test_raster_order_swizzle_2():
    assert m.raster_order(3, 2, 2) == [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (2, 1)]


@pytest.mark.parametrize("mt,nt,s", [(7, 5, 1), (7, 5, 2), (8, 8, 4), (5, 9, 8)])
def test_raster_order_is_a_permutation(mt, nt, s):
    o = m.raster_order(mt, nt, s)
    assert sorted(o) == [(a, b) for a in range(mt) for b in range(nt)]


def test_raster_order_rejects_bad_swizzle():
    with pytest.raises(ValueError):
        m.raster_order(2, 2, 0)


def test_swizzle_reduces_l2_traffic():
    a = b = 32 * 1024
    row = m.wave_unique_bytes(m.raster_order(32, 32, 1), 132, a, b)
    sw = m.wave_unique_bytes(m.raster_order(32, 32, 8), 132, a, b)
    assert sw < row * 0.8
    assert m.wave_unique_bytes([(0, 0), (0, 1), (1, 0)], 2, 10, 1) == (10 + 2) + (10 + 1)


def test_persistent_static_round_robin():
    a = m.persistent_static(10, 4)
    assert a == [[0, 4, 8], [1, 5, 9], [2, 6], [3, 7]]


def test_stream_k_small_example():
    a = m.stream_k(3, 4, 2)                  # 12 iterations, 6 each
    assert a == [[(0, 0, 4), (1, 0, 2)], [(1, 2, 4), (2, 0, 4)]]
    assert m.fixup_tiles(a) == {1: 2}
    assert m.makespan(a) == 6


@pytest.mark.parametrize("tiles,k,sms", [(200, 64, 132), (7, 3, 5), (1, 100, 8), (13, 1, 4)])
def test_stream_k_covers_every_iteration_once_and_balances(tiles, k, sms):
    a = m.stream_k(tiles, k, sms)
    cnt = Counter()
    for segs in a:
        for t, b, e in segs:
            assert 0 <= b < e <= k
            for kk in range(b, e):
                cnt[(t, kk)] += 1
    assert len(cnt) == tiles * k and set(cnt.values()) == {1}
    loads = [sum(e - b for _, b, e in segs) for segs in a]
    assert max(loads) - min(loads) <= 1


def test_stream_k_beats_data_parallel_on_wave_tail():
    tiles, k, sms = 200, 64, 132
    dp = m.wave_stats(tiles, sms)["waves"] * k
    assert m.makespan(m.stream_k(tiles, k, sms)) < dp * 0.8


def test_makespan_empty():
    assert m.makespan([]) == 0
