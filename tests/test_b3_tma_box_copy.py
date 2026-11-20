import numpy as np
import pytest

from drill import load

m = load("b3_tma_box_copy")


def G(H=10, W=70):
    return np.arange(1, H * W + 1, dtype=np.int64).reshape(H, W)


def test_plain_box_is_row_major_copy():
    g = G()
    s = m.tma_load_2d(g, 2, 3, 4, 8)
    assert s.shape == (32,)
    assert np.array_equal(s.reshape(4, 8), g[2:6, 3:11])


def test_out_of_bounds_is_zero_filled():
    g = G()
    s = m.tma_load_2d(g, 8, 66, 4, 8).reshape(4, 8)
    assert np.array_equal(s[:2, :4], g[8:10, 66:70])
    assert (s[2:, :] == 0).all() and (s[:, 4:] == 0).all()


def test_negative_coordinates_are_out_of_bounds():
    g = G()
    s = m.tma_load_2d(g, -1, -2, 3, 4).reshape(3, 4)
    assert (s[0] == 0).all() and (s[:, :2] == 0).all()
    assert np.array_equal(s[1:, 2:], g[0:2, 0:2])


def test_swizzle_offsets():
    assert m.swizzle_byte_offset(0, 128) == 0
    assert m.swizzle_byte_offset(128, 128) == 128 + 16           # row 1, chunk 0 -> chunk 1
    assert m.swizzle_byte_offset(7 * 128 + 16 * 2 + 6, 128) == 7 * 128 + 16 * (2 ^ 7) + 6
    assert m.swizzle_byte_offset(5 * 128, 64) == 5 * 128 + 16 * (5 % 4)
    assert m.swizzle_byte_offset(3 * 128, 32) == 3 * 128 + 16 * (3 % 2)
    assert m.swizzle_byte_offset(1000, 0) == 1000


@pytest.mark.parametrize("swz,cols", [(128, 64), (64, 32), (32, 16)])
def test_swizzled_box_reads_back_through_same_rule(swz, cols):
    g = G(12, 80)
    s = m.tma_load_2d(g, 1, 5, 8, cols, swizzle_bytes=swz)
    assert sorted(s.tolist()) == sorted(g[1:9, 5:5 + cols].ravel().tolist())
    for r in range(8):
        for c in range(cols):
            assert m.smem_read(s, r, c, cols, swz) == g[1 + r, 5 + c]


def test_128b_swizzle_actually_moves_data():
    g = G(8, 64)
    s = m.tma_load_2d(g, 0, 0, 8, 64, swizzle_bytes=128)
    assert s[64] != g[1, 0]          # row 1 chunk 0 is not at its row-major place
    assert s[64 + 8] == g[1, 0]


def test_invalid_swizzle_configs_rejected():
    g = G()
    with pytest.raises(ValueError):
        m.tma_load_2d(g, 0, 0, 2, 8, swizzle_bytes=16)
    with pytest.raises(ValueError):
        m.tma_load_2d(g, 0, 0, 2, 128, swizzle_bytes=128)   # 256 B rows > 128 B span


def test_tx_bytes_counts_whole_box_even_out_of_bounds():
    assert m.tx_bytes(64, 64) == 8192
    assert m.tx_bytes(128, 64, elem_bytes=1) == 8192


def test_tile_boxes_cover_with_tails():
    assert m.tile_boxes(100, 70, 64, 64) == [(0, 0), (0, 64), (64, 0), (64, 64)]
    assert m.tile_boxes(128, 64, 64, 64) == [(0, 0), (64, 0)]


def test_multicast_masks_2x2():
    masks = m.multicast_masks(2, 2)
    # ranks: (0,0)=0 (1,0)=1 (0,1)=2 (1,1)=3
    assert masks[0] == (0b0101, 0b0011)
    assert masks[3] == (0b1010, 0b1100)


def test_multicast_masks_2x1_and_1x1():
    assert m.multicast_masks(2, 1) == {0: (0b01, 0b11), 1: (0b10, 0b11)}
    assert m.multicast_masks(1, 1) == {0: (1, 1)}


def test_l2_traffic_with_and_without_multicast():
    a = b = 128 * 64 * 2
    base = m.l2_bytes_per_kstep(8, 8, a, b)
    assert base == 64 * (a + b)
    assert m.l2_bytes_per_kstep(8, 8, a, b, cm=2, cn=1) == 64 * (a + b / 2)
    assert m.l2_bytes_per_kstep(8, 8, a, b, cm=2, cn=2) == base / 2
    with pytest.raises(ValueError):
        m.l2_bytes_per_kstep(3, 8, a, b, cm=2)
