import pytest

from drill import load

m = load("c1_tmem_accumulators")


def test_two_256_column_accumulators_fill_tmem():
    a = m.TmemAllocator()
    assert a.alloc(256) == 0
    assert a.alloc(256) == 256
    assert a.used == 512
    assert a.alloc(32) is None


def test_alignment_first_fit():
    a = m.TmemAllocator()
    assert a.alloc(32) == 0
    assert a.alloc(64) == 64          # 32..63 is free but not 64-aligned
    assert a.alloc(32) == 32
    assert a.alloc(128) == 128


def test_free_and_reuse():
    a = m.TmemAllocator()
    x = a.alloc(128)
    a.alloc(128)
    a.free(x, 128)
    assert a.used == 128
    assert a.alloc(128) == x


@pytest.mark.parametrize("bad", [0, 16, 48, 100, 1024])
def test_bad_sizes_rejected(bad):
    with pytest.raises(ValueError):
        m.TmemAllocator().alloc(bad)


def test_bad_free_rejected():
    a = m.TmemAllocator()
    a.alloc(64)
    with pytest.raises(ValueError):
        a.free(0, 32)
    a.free(0, 64)
    with pytest.raises(ValueError):
        a.free(0, 64)


def test_failed_alloc_changes_nothing():
    a = m.TmemAllocator(capacity=128)
    a.alloc(64)
    assert a.alloc(128) is None
    assert a.used == 64
    assert a.alloc(64) == 64


def test_capacity_576_for_rubin():
    a = m.TmemAllocator(capacity=576)
    assert a.alloc(512) == 0
    assert a.alloc(32) == 512
    assert a.alloc(32) == 544
    assert a.alloc(32) is None


def test_acc_columns_and_buffers():
    assert m.acc_columns(256) == 256
    assert m.acc_columns(128, acc_bits=16) == 64
    assert m.max_acc_buffers(256) == 2
    assert m.max_acc_buffers(128) == 4


def test_single_accumulator_serializes():
    assert m.acc_pipeline(4, 10, 6, 1) == 4 * 16


def test_double_buffer_overlaps_epilogue():
    assert m.acc_pipeline(4, 10, 6, 2) == 46
    assert m.acc_pipeline(4, 10, 15, 2) == 10 + 4 * 15     # epilogue bound


def test_more_buffers_never_slower_and_zero_tiles():
    for ML, EP in [(10, 6), (4, 9), (7, 7)]:
        ts = [m.acc_pipeline(12, ML, EP, b) for b in (1, 2, 3, 4)]
        assert ts == sorted(ts, reverse=True)
    assert m.acc_pipeline(0, 5, 5, 2) == 0
    with pytest.raises(ValueError):
        m.acc_pipeline(3, 1, 1, 0)


def test_2cta_mma_intensity():
    assert m.per_sm_intensity(128, 256) == pytest.approx(85.33, rel=1e-3)
    assert m.per_sm_intensity(256, 256, cta_pair=True) == pytest.approx(128)
    assert m.per_sm_intensity(256, 256, cta_pair=True) >= 64     # B200 needs 64
