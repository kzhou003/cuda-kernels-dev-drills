import numpy as np
import pytest

from drill import load

m = load("d3_fa4_softmax_tricks")


def ref(Q, K, V, scale):
    S = scale * Q @ K.T
    P = np.exp(S - S.max(axis=1, keepdims=True))
    return (P @ V) / P.sum(axis=1, keepdims=True)


def test_ex2_exact_at_integers():
    xs = np.arange(-20, 21, dtype=np.float32)
    assert np.array_equal(m.ex2_emulation(xs), np.exp2(xs).astype(np.float32))


def test_ex2_relative_error_is_enough_for_bf16():
    x = np.linspace(-30, 10, 20001, dtype=np.float32)
    y = m.ex2_emulation(x)
    assert y.dtype == np.float32
    rel = np.abs(y.astype(np.float64) - np.exp2(x.astype(np.float64))) / np.exp2(x.astype(np.float64))
    assert rel.max() < 2e-4            # bf16 keeps 8 mantissa bits: 2^-9 ~ 2e-3
    assert rel.max() > 1e-6            # it is an approximation, not np.exp2


def test_ex2_negative_fraction_uses_floor():
    # -0.5 = -1 + 0.5 -> 2^-1 * poly(0.5)
    y = m.ex2_emulation(np.float32(-0.5))
    assert abs(float(y) - 2 ** -0.5) < 1e-4
    assert np.isscalar(y) or np.ndim(y) == 0


def test_ex2_clamps_very_negative_inputs():
    y = m.ex2_emulation(np.array([-1000.0, -127.0], dtype=np.float32))
    assert np.all(np.isfinite(y)) and np.all(y >= 0) and np.all(y < 1e-37)


@pytest.mark.parametrize("threshold", [0.0, 2.0, 8.0])
def test_conditional_rescale_is_still_exact(threshold):
    r = np.random.default_rng(0)
    Q, K, V = r.standard_normal((8, 16)), r.standard_normal((200, 16)), r.standard_normal((200, 4))
    O, _ = m.softmax_attention_cond_rescale(Q, K, V, 0.5, 16, threshold)
    assert np.allclose(O, ref(Q, K, V, 0.5))


def test_threshold_reduces_rescales():
    r = np.random.default_rng(1)
    Q, K, V = r.standard_normal((8, 16)), r.standard_normal((512, 16)), r.standard_normal((512, 4))
    counts = [m.softmax_attention_cond_rescale(Q, K, V, 0.5, 32, t)[1] for t in (0.0, 1.0, 8.0)]
    assert counts[0] > counts[1] >= counts[2]
    assert counts[2] == 0


def test_large_jump_still_rescales_and_counts_per_row():
    Q = np.eye(2)
    K = np.array([[0.0, 0.0], [0.0, 0.0], [100.0, 0.0], [0.0, 0.0]])
    V = np.arange(8.0).reshape(4, 2)
    O, n = m.softmax_attention_cond_rescale(Q, K, V, 1.0, 2, 8.0)
    assert n == 1                              # only row 0 sees the jump of 100*log2(e)
    assert np.allclose(O, ref(Q, K, V, 1.0))


def test_increasing_scores_without_threshold_rescale_every_block():
    Q = np.ones((1, 1))
    K = np.arange(8.0)[:, None]
    V = np.ones((8, 1))
    _, n = m.softmax_attention_cond_rescale(Q, K, V, 1.0, 1, 0.0)
    assert n == 7


def test_exp_to_mma_ratio_per_generation():
    assert m.exp_to_mma_ratio(2048, 16, 128) == pytest.approx(0.25)
    assert m.exp_to_mma_ratio(4096, 16, 128) == pytest.approx(0.5)
    assert m.exp_to_mma_ratio(8192, 16, 128) == pytest.approx(1.0)


def test_balanced_fraction():
    f = m.balanced_emulation_fraction(16, 128, 6)
    assert f == pytest.approx(128 / (128 + 96))
    assert (1 - f) / 16 == pytest.approx(f * 6 / 128)
