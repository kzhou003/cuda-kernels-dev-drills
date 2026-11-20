import numpy as np
import pytest

from drill import load

m = load("a2_mma_fragments")


def test_lane_zero_and_lane_five_coordinates():
    assert m.a_coords(0) == [(0, 0), (0, 1), (8, 0), (8, 1), (0, 8), (0, 9), (8, 8), (8, 9)]
    assert m.a_coords(5) == [(1, 2), (1, 3), (9, 2), (9, 3), (1, 10), (1, 11), (9, 10), (9, 11)]
    assert m.b_coords(5) == [(2, 1), (3, 1), (10, 1), (11, 1)]
    assert m.c_coords(5) == [(1, 2), (1, 3), (9, 2), (9, 3)]


@pytest.mark.parametrize("fn,shape,count", [("a_coords", (16, 16), 8), ("b_coords", (16, 8), 4), ("c_coords", (16, 8), 4)])
def test_each_element_owned_by_exactly_one_lane(fn, shape, count):
    seen = {}
    for lane in range(32):
        cs = getattr(m, fn)(lane)
        assert len(cs) == count
        for c in cs:
            assert c not in seen
            seen[c] = lane
    assert len(seen) == shape[0] * shape[1]


def test_mma_matches_numpy_matmul():
    rng = np.random.default_rng(0)
    A, B, C = rng.standard_normal((16, 16)), rng.standard_normal((16, 8)), rng.standard_normal((16, 8))
    a = np.array([[A[r, k] for r, k in m.a_coords(l)] for l in range(32)])
    b = np.array([[B[k, n] for k, n in m.b_coords(l)] for l in range(32)])
    c = np.array([[C[r, n] for r, n in m.c_coords(l)] for l in range(32)])
    d = m.mma_m16n8k16(a, b, c)
    D = A @ B + C
    for l in range(32):
        assert np.allclose(d[l], [D[r, n] for r, n in m.c_coords(l)])


def test_c_layout_rows_match_a_layout_rows():
    # the FlashAttention trick: S (C layout) feeds P@V (A layout) without shuffles
    for l in range(32):
        a_rows = {r for r, _ in m.a_coords(l)}
        c_rows = {r for r, _ in m.c_coords(l)}
        assert a_rows == c_rows


def test_ldmatrix_x1_distributes_rows_to_quads():
    smem = np.arange(64, dtype=np.int64)
    out = m.ldmatrix(smem, [8 * r for r in range(8)] + [0] * 24, 1)
    assert out.shape == (32, 2)
    assert list(out[0]) == [0, 1]
    assert list(out[3]) == [6, 7]
    assert list(out[4]) == [8, 9]       # lane 4 -> row 1
    assert list(out[31]) == [62, 63]


def test_ldmatrix_rejects_bad_num():
    with pytest.raises(ValueError):
        m.ldmatrix(np.zeros(64), [0] * 32, 3)


@pytest.mark.parametrize("ld", [16, 24, 40])
def test_ldmatrix_x4_with_a_addresses_produces_a_fragments(ld):
    rng = np.random.default_rng(1)
    base = 3 * ld + 8
    smem = rng.integers(0, 1000, size=base + 16 * ld + 16)
    A = np.array([[smem[base + r * ld + c] for c in range(16)] for r in range(16)])
    out = m.ldmatrix(smem, m.a_tile_row_addrs(base, ld), 4)
    for l in range(32):
        assert list(out[l]) == [A[r, k] for r, k in m.a_coords(l)]


@pytest.mark.parametrize("ld", [16, 40])
def test_ldmatrix_x2_with_b_addresses_produces_b_fragments(ld):
    rng = np.random.default_rng(2)
    base = 16
    smem = rng.integers(0, 1000, size=base + 8 * ld + 16)
    Bt = np.array([[smem[base + n * ld + k] for k in range(16)] for n in range(8)])   # N x K
    out = m.ldmatrix(smem, m.b_tile_row_addrs(base, ld), 2)
    for l in range(32):
        assert list(out[l]) == [Bt[n, k] for k, n in m.b_coords(l)]


def test_full_warp_gemm_from_smem():
    # A 16x16 and B^T 8x16 in "smem" -> ldmatrix -> mma == A @ B
    rng = np.random.default_rng(3)
    A = rng.standard_normal((16, 16))
    Bt = rng.standard_normal((8, 16))
    smem = np.concatenate([A.ravel(), Bt.ravel()])
    a = m.ldmatrix(smem, m.a_tile_row_addrs(0, 16), 4)
    b = m.ldmatrix(smem, m.b_tile_row_addrs(256, 16), 2)
    d = m.mma_m16n8k16(a, b, np.zeros((32, 4)))
    D = A @ Bt.T
    for l in range(32):
        assert np.allclose(d[l], [D[r, n] for r, n in m.c_coords(l)])
