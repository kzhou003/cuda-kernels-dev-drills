import numpy as np
import pytest

from drill import load

m = load("d1_online_softmax")


def ref_attention(Q, K, V, scale, mask=None):
    S = scale * Q @ K.T
    if mask is not None:
        S = np.where(mask, S, -np.inf)
    mx = S.max(axis=1, keepdims=True)
    ok = np.isfinite(mx[:, 0])
    mx = np.where(np.isfinite(mx), mx, 0)
    P = np.exp(S - mx)
    l = P.sum(axis=1)
    O = np.where(ok[:, None], (P @ V) / np.where(l > 0, l, 1)[:, None], 0)
    lse = np.where(ok, mx[:, 0] + np.log(np.where(l > 0, l, 1)), -np.inf)
    return O, lse


def data(nq=5, nk=37, d=8, seed=0):
    r = np.random.default_rng(seed)
    return r.standard_normal((nq, d)), r.standard_normal((nk, d)), r.standard_normal((nk, 6))


@pytest.mark.parametrize("block", [1, 3, 8, 100])
def test_stats_match_direct(block):
    x = np.array([3.0, -1.0, 7.5, 2.0, 7.5, -20.0, 0.1])
    mm, l = m.online_softmax_stats(x, block)
    assert mm == 7.5
    assert l == pytest.approx(np.exp(x - 7.5).sum())


def test_stats_empty_and_all_neg_inf():
    assert m.online_softmax_stats([], 4) == (-np.inf, 0.0)
    assert m.online_softmax_stats([-np.inf] * 5, 2) == (-np.inf, 0.0)
    mm, l = m.online_softmax_stats([-np.inf, -np.inf, 1.0], 2)
    assert mm == 1.0 and l == pytest.approx(1.0)


def test_stats_large_values_do_not_overflow():
    mm, l = m.online_softmax_stats([1000.0, 999.0, 1001.0], 1)
    assert np.isfinite(l) and l == pytest.approx(np.exp([-1, -2, 0]).sum())


@pytest.mark.parametrize("block", [1, 4, 16, 64])
def test_online_attention_matches_reference(block):
    Q, K, V = data()
    O, lse = m.online_attention(Q, K, V, 0.3, block)
    Or, lr = ref_attention(Q, K, V, 0.3)
    assert np.allclose(O, Or) and np.allclose(lse, lr)


def test_fully_masked_rows_give_zero_and_neg_inf():
    Q, K, V = data(nq=3, nk=10)
    mask = np.ones((3, 10), bool)
    mask[1] = False
    mask[2, :7] = False
    O, lse = m.online_attention(Q, K, V, 0.5, 4, mask)
    assert not np.isnan(O).any()
    assert (O[1] == 0).all() and lse[1] == -np.inf
    Or, lr = ref_attention(Q, K, V, 0.5, mask)
    assert np.allclose(O, Or) and np.allclose(lse[[0, 2]], lr[[0, 2]])


def test_merge_two_halves_equals_full():
    Q, K, V = data()
    oa, la = m.online_attention(Q, K[:20], V[:20], 0.3, 8)
    ob, lb = m.online_attention(Q, K[20:], V[20:], 0.3, 8)
    o, l = m.merge_states(oa, la, ob, lb)
    Or, lr = ref_attention(Q, K, V, 0.3)
    assert np.allclose(o, Or) and np.allclose(l, lr)


def test_merge_with_empty_side_is_identity():
    Q, K, V = data(nq=2)
    oa, la = m.online_attention(Q, K, V, 0.3, 8)
    empty_o, empty_l = np.zeros_like(oa), np.full(2, -np.inf)
    o, l = m.merge_states(oa, la, empty_o, empty_l)
    assert np.allclose(o, oa) and np.allclose(l, la)
    o, l = m.merge_states(empty_o, empty_l, empty_o, empty_l)
    assert (o == 0).all() and (l == -np.inf).all() and not np.isnan(o).any()


@pytest.mark.parametrize("splits", [1, 2, 5, 37, 50])
def test_split_kv_matches_reference(splits):
    Q, K, V = data()
    o, l = m.split_kv_attention(Q, K, V, 0.3, splits)
    Or, lr = ref_attention(Q, K, V, 0.3)
    assert np.allclose(o, Or) and np.allclose(l, lr)


def test_split_kv_with_mask_where_some_splits_are_fully_masked():
    Q, K, V = data(nq=4, nk=40)
    mask = np.zeros((4, 40), bool)
    mask[:, 30:] = True
    mask[3] = False
    o, l = m.split_kv_attention(Q, K, V, 0.3, 4, mask)
    Or, lr = ref_attention(Q, K, V, 0.3, mask)
    assert not np.isnan(o).any()
    assert np.allclose(o, Or) and np.allclose(l[:3], lr[:3]) and l[3] == -np.inf
