import numpy as np
import pytest

from drill import load

m = load("c2_block_scaling")


def test_e2m1_exact_grid_points_and_sign():
    g = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    assert list(m.quantize_e2m1(g)) == g
    assert list(m.quantize_e2m1([-g_ for g_ in g])) == [-g_ for g_ in g]


def test_e2m1_ties_go_to_even_mantissa():
    assert list(m.quantize_e2m1([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])) == [0, 1, 1, 2, 2, 4, 4]


def test_e2m1_nearest_and_saturation():
    assert list(m.quantize_e2m1([0.3, 2.6, 4.9, 5.1, 100, -7])) == [0.5, 3, 4, 6, 6, -6]


def test_mx_e4m3_scales_are_powers_of_two():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(128) * 10
    q, s = m.mx_quantize(x, 32, "e4m3")
    assert len(s) == 4
    assert np.all(np.log2(s) == np.round(np.log2(s)))
    blk = x.reshape(4, 32)
    for i in range(4):
        assert s[i] == 2.0 ** (np.floor(np.log2(np.abs(blk[i]).max())) - 8)
    assert np.all(np.abs(q) <= 448)
    rel = np.abs(m.dequantize(q, s, 32) - x) / np.abs(x).max()
    assert rel.max() < 0.07


def test_mx_e2m1_known_block():
    x = np.array([7.0, 1.0, -0.4, 0.0] * 8)
    q, s = m.mx_quantize(x, 32, "e2m1")
    assert s[0] == 1.0                      # floor(log2 7) - emax(2) = 0
    assert list(q[:4]) == [6.0, 1.0, -0.5, 0.0]   # 7 saturates to 6


def test_mx_zero_block_and_errors():
    q, s = m.mx_quantize(np.zeros(32), 32, "e2m1")
    assert s[0] == 1.0 and not q.any()
    with pytest.raises(ValueError):
        m.mx_quantize(np.zeros(33))
    with pytest.raises(ValueError):
        m.mx_quantize(np.zeros(32), elem="e5m2")


def test_nvfp4_round_trip_better_than_mxfp4():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(4096) * np.repeat(rng.uniform(0.1, 3, 128), 32)
    q, s, g = m.nvfp4_quantize(x)
    nv = np.abs(m.dequantize(q, s, 16, g) - x).mean()
    q2, s2 = m.mx_quantize(x, 32, "e2m1")
    mx = np.abs(m.dequantize(q2, s2, 32) - x).mean()
    assert nv < mx
    assert set(np.unique(np.abs(q))) <= set(m.E2M1_GRID)


def test_nvfp4_global_scale_and_zero_block():
    x = np.concatenate([np.zeros(16), np.linspace(-3, 3, 16)])
    q, s, g = m.nvfp4_quantize(x)
    assert g == pytest.approx(3 / (448 * 6))
    assert s[0] == 0 and not q[:16].any()
    assert s[1] == pytest.approx(448.0)
    q, s, g = m.nvfp4_quantize(np.zeros(16))
    assert g == 1.0


def test_block_scaled_dot_matches_dequantized_dot():
    rng = np.random.default_rng(2)
    a, b = rng.standard_normal(256), rng.standard_normal(256)
    qa, sa = m.mx_quantize(a, 32, "e4m3")
    qb, sb = m.mx_quantize(b, 32, "e4m3")
    ref = float(np.dot(m.dequantize(qa, sa, 32), m.dequantize(qb, sb, 32)))
    assert m.block_scaled_dot(qa, sa, qb, sb, 32) == pytest.approx(ref, rel=1e-12)


def test_promotion_fixes_swamping():
    rng = np.random.default_rng(3)
    a = rng.uniform(0.5, 1.5, 8192)
    b = rng.uniform(0.5, 1.5, 8192)
    exact = float(np.dot(a, b))
    never = m.promoted_dot(a, b, 13)
    every128 = m.promoted_dot(a, b, 13, 128)
    assert abs(never - exact) / exact > 5e-4
    assert abs(every128 - exact) / exact < 5e-6
    # 10-bit accumulator without promotion stops growing: ~47% error
    assert abs(m.promoted_dot(a, b, 10) - exact) / exact > 0.3
    assert m.promoted_dot(a, b, 52) == pytest.approx(exact, rel=1e-12)


def test_promotion_boundaries_counted_from_one():
    # with a huge mantissa nothing is lost; result independent of promote_every
    a = np.arange(10.0)
    assert m.promoted_dot(a, a, 52, 3) == pytest.approx(float(a @ a))
