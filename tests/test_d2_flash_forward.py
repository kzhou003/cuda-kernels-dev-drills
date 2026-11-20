import numpy as np
import pytest

from drill import load

m = load("d2_flash_forward")


def ref(Q, K, V, scale, causal):
    S = scale * Q @ K.T
    nq, nk = S.shape
    if causal:
        S = np.where(np.arange(nk)[None, :] <= np.arange(nq)[:, None] + (nk - nq), S, -np.inf)
    mx = S.max(axis=1, keepdims=True)
    ok = np.isfinite(mx[:, 0])
    mx = np.where(np.isfinite(mx), mx, 0)
    P = np.exp(S - mx)
    l = P.sum(axis=1)
    O = np.where(ok[:, None], (P @ V) / np.where(l > 0, l, 1)[:, None], 0)
    return O, np.where(ok, mx[:, 0] + np.log(np.where(l > 0, l, 1)), -np.inf)


def data(nq, nk, d=16, seed=0):
    r = np.random.default_rng(seed)
    return r.standard_normal((nq, d)), r.standard_normal((nk, d)), r.standard_normal((nk, d))


@pytest.mark.parametrize("nq,nk,Br,Bc", [(64, 64, 16, 16), (50, 70, 16, 32), (33, 33, 8, 4), (7, 129, 64, 64)])
@pytest.mark.parametrize("causal", [False, True])
def test_matches_reference(nq, nk, Br, Bc, causal):
    Q, K, V = data(nq, nk)
    O, lse, _ = m.flash_forward(Q, K, V, 0.25, Br, Bc, causal)
    Or, lr = ref(Q, K, V, 0.25, causal)
    assert np.allclose(O, Or)
    fin = np.isfinite(lr)
    assert np.allclose(lse[fin], lr[fin]) and (lse[~fin] == -np.inf).all()


def test_noncausal_computes_every_block():
    Q, K, V = data(64, 64)
    _, _, st = m.flash_forward(Q, K, V, 0.25, 16, 16)
    assert st == {"computed": 16, "skipped": 0}


def test_causal_square_skips_upper_triangle():
    Q, K, V = data(64, 64)
    _, _, st = m.flash_forward(Q, K, V, 0.25, 16, 16, causal=True)
    assert st == {"computed": 10, "skipped": 6}
    assert m.causal_blocks(64, 64, 16, 16) == 10


def test_causal_with_more_queries_than_keys_masks_whole_rows():
    Q, K, V = data(40, 10)
    O, lse, st = m.flash_forward(Q, K, V, 0.25, 8, 8, causal=True)
    assert not np.isnan(O).any()
    assert (lse[:30] == -np.inf).all() and (O[:30] == 0).all()
    assert st["computed"] == m.causal_blocks(40, 10, 8, 8)


@pytest.mark.parametrize("nq,nk,Br,Bc", [(128, 128, 32, 64), (100, 300, 64, 32), (1, 1000, 64, 128)])
def test_causal_blocks_matches_forward_stats(nq, nk, Br, Bc):
    Q, K, V = data(nq, nk, d=4)
    _, _, st = m.flash_forward(Q, K, V, 0.25, Br, Bc, causal=True)
    assert st["computed"] == m.causal_blocks(nq, nk, Br, Bc)
    assert st["computed"] + st["skipped"] == -(-nq // Br) * -(-nk // Bc)


def test_causal_saves_about_half_for_long_sequences():
    total = (4096 // 128) ** 2
    assert m.causal_blocks(4096, 4096, 128, 128) / total == pytest.approx(0.5, abs=0.02)


def test_hbm_bytes_formulas():
    N, d = 4096, 128
    assert m.hbm_bytes(N, d, 128, 128, "fa2") == (2 + 2 * 32) * N * d * 2
    assert m.hbm_bytes(N, d, 128, 128, "fa1") == (2 + 3 * 32) * N * d * 2
    assert m.hbm_bytes(1000, 64, 128, 64, "fa1") == (2 + 3 * 16) * 1000 * 64 * 2
    with pytest.raises(ValueError):
        m.hbm_bytes(N, d, 128, 128, "fa3")


def test_fa2_moves_fewer_bytes_than_fa1():
    for N in (1024, 8192):
        assert m.hbm_bytes(N, 64, 128, 128, "fa2") < m.hbm_bytes(N, 64, 128, 128, "fa1")
