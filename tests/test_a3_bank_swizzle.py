import pytest

from drill import load

m = load("a3_bank_swizzle")


def test_consecutive_words_are_conflict_free():
    assert m.wavefronts_32bit([4 * i for i in range(32)]) == 1


def test_broadcast_is_not_a_conflict():
    assert m.wavefronts_32bit([128] * 32) == 1
    assert m.wavefronts_32bit([]) == 0


def test_column_read_of_32x32_float_tile_is_32_way():
    ld = 32
    assert m.wavefronts_32bit([(r * ld + 5) * 4 for r in range(32)]) == 32


def test_padding_one_float_fixes_column_read():
    assert m.wavefronts_32bit([m.padded_offset(r, 5, 32, 1) * 4 for r in range(32)]) == 1


def test_stride_two_words_is_two_way():
    assert m.wavefronts_32bit([8 * i for i in range(32)]) == 2


def test_ldmatrix_on_128_byte_rows_is_8_way():
    # bf16 tile with 64 elements per row: column block 0 of rows 0..7
    addrs = [r * 128 for r in range(8)]
    assert m.ldmatrix_wavefronts(addrs) == 8


def test_ldmatrix_contiguous_matrix_is_one_wavefront():
    assert m.ldmatrix_wavefronts([16 * r for r in range(8)]) == 1
    assert m.ldmatrix_wavefronts([16 * r for r in range(32)]) == 4     # x4: one per matrix


def test_ldmatrix_validates_alignment():
    with pytest.raises(ValueError):
        m.ldmatrix_wavefronts([8 * r for r in range(8)])
    with pytest.raises(ValueError):
        m.ldmatrix_wavefronts([0, 16, 32])


def test_swizzle_known_values():
    assert m.swizzle(0, 3, 3, 3) == 0
    assert m.swizzle(64, 3, 3, 3) == 64 + 8        # row 1, chunk 0 -> chunk 1
    assert m.swizzle(64 * 3 + 8 * 2 + 5, 3, 3, 3) == 64 * 3 + 8 * (2 ^ 3) + 5
    assert m.swizzle(64 * 9, 3, 3, 3) == 64 * 9 + 8   # row 9 behaves like row 1


def test_swizzle_is_a_permutation_and_its_own_inverse():
    for B, M, S in [(3, 3, 3), (2, 3, 3), (3, 4, 3), (1, 4, 3)]:
        span = 1 << (B + M + S)
        img = [m.swizzle(o, B, M, S) for o in range(4 * span)]
        assert sorted(img) == list(range(4 * span))
        assert all(m.swizzle(m.swizzle(o, B, M, S), B, M, S) == o for o in range(4 * span))


def test_swizzle_keeps_16_byte_vectors_contiguous():
    for o in range(0, 4096, 8):
        base = m.swizzle(o, 3, 3, 3)
        assert [m.swizzle(o + i, 3, 3, 3) for i in range(8)] == [base + i for i in range(8)]


@pytest.mark.parametrize("col_block", range(8))
def test_swizzle_makes_ldmatrix_conflict_free(col_block):
    addrs = [2 * m.swizzled_offset(r, 8 * col_block, 64, 3, 3, 3) for r in range(8)]
    assert m.ldmatrix_wavefronts(addrs) == 1


def test_swizzle_also_conflict_free_for_row_reads():
    # one row (8 chunks) read as one matrix of 8 "rows" of 16 B
    addrs = [2 * m.swizzled_offset(5, 8 * c, 64, 3, 3, 3) for c in range(8)]
    assert m.ldmatrix_wavefronts(addrs) == 1
