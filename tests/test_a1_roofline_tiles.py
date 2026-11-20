import pytest

from drill import load

m = load("a1_roofline_tiles")
G = m.GPUS


def test_machine_balance_h100_is_about_295():
    assert m.machine_balance(G["H100"]) == pytest.approx(295, rel=0.02)


def test_smem_balance_doubles_every_datacenter_generation():
    vals = [m.smem_balance(G[k]) for k in ("V100", "A100", "H100", "B200")]
    assert vals == [8, 16, 32, 64]
    assert m.smem_balance(G["RTX5090"]) == 4


def test_tile_intensity_matches_area_over_perimeter():
    assert m.tile_intensity(64, 64) == pytest.approx(32)
    assert m.tile_intensity(64, 256) == pytest.approx(51.2)
    assert m.tile_intensity(128, 256) == pytest.approx(85.33, rel=1e-3)
    # FP8 operands: half the bytes, twice the intensity
    assert m.tile_intensity(128, 128, elem_bytes=1) == pytest.approx(2 * m.tile_intensity(128, 128))


def test_gemm_intensity_counts_output_bytes():
    assert m.gemm_intensity(4096, 4096, 4096) == pytest.approx(2 * 4096 ** 3 / (2 * 2 * 4096 ** 2 + 2 * 4096 ** 2))
    # decode-shaped GEMV is hopelessly memory bound
    assert m.gemm_intensity(1, 4096, 4096) < 1.01


def test_attainable_is_min_of_roof_and_slope():
    h = G["H100"]
    assert m.attainable_tflops(h, 1.0) == pytest.approx(3.35)
    assert m.attainable_tflops(h, 10_000) == pytest.approx(h.peak_tflops)


def test_smallest_feeding_tile_per_generation():
    cands = [(8, 8), (16, 16), (32, 32), (64, 64), (64, 128), (128, 64), (64, 256), (128, 128), (128, 256), (256, 256)]
    assert m.smallest_feeding_tile(G["RTX5090"], cands) == (8, 8)
    assert m.smallest_feeding_tile(G["A100"], cands) == (32, 32)
    assert m.smallest_feeding_tile(G["H100"], cands) == (64, 64)
    assert m.smallest_feeding_tile(G["B200"], cands) == (128, 128)
    # ties on area break toward smaller M
    assert m.smallest_feeding_tile(G["H100"], [(128, 64), (64, 128)]) == (64, 128)
    assert m.smallest_feeding_tile(G["B200"], [(16, 16), (32, 32)]) is None


def test_accumulator_registers():
    assert m.accumulator_regs_per_thread(64, 256, 128) == 128     # one wgmma 64x256 per warpgroup
    assert m.accumulator_regs_per_thread(128, 256, 128) == 256    # does not fit: > 255
    assert m.accumulator_regs_per_thread(16, 8, 32) == 4          # one mma.sync m16n8 tile per warp
    assert m.accumulator_regs_per_thread(10, 10, 32) == 4         # rounds up


def test_ctas_per_sm_register_granularity():
    a = G["A100"]
    # 256 threads x 32 regs = 8192 regs -> 8 CTAs by registers, 8 by threads
    assert m.ctas_per_sm(a, 256, 32, 0) == 8
    # 33 regs rounds up to 40 per thread: 65536 // (8 warps * 1280) = 6
    assert m.ctas_per_sm(a, 256, 33, 0) == 6


def test_ctas_per_sm_smem_and_thread_limits():
    g = G["RTX5090"]
    assert m.ctas_per_sm(g, 128, 64, 48 * 1024) == 2
    assert m.ctas_per_sm(g, 128, 64, 100 * 1024) == 0          # over 99 KB per CTA on sm120
    assert m.ctas_per_sm(g, 1024, 16, 0) == 1                    # 1536 threads per SM
    assert m.ctas_per_sm(g, 32, 8, 0) == 32                      # CTA-count cap
    assert m.ctas_per_sm(G["H100"], 128, 256, 0) == 0            # > 255 registers
