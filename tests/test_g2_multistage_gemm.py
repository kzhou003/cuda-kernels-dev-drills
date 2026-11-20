import pytest

from drill import load
from tests._cuda import requires_cuda, torch

pytestmark = requires_cuda
m = load("g2_multistage_gemm")

SHAPES = [(128, 128, 32), (256, 384, 64), (100, 70, 96), (1, 1, 32), (513, 257, 1024), (2048, 2048, 512)]


@pytest.mark.parametrize("stages", [2, 3, 4])
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_matches_fp32_reference(M, N, K, stages):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    ref = A.float() @ B.float().T
    out = m.gemm_pipelined(A, B, stages)
    assert out.dtype == torch.float32 and out.shape == (M, N)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-2 * (K / 32) ** 0.5)


def test_identity_picks_rows_exactly():
    A = torch.randn(64, 64, device="cuda").bfloat16()
    I = torch.eye(64, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(m.gemm_pipelined(A, I), A.float())


def test_structured_values_catch_layout_bugs():
    # A[i, k] = i, B[n, k] = (k == n): C[i, n] = i for n < K. A wrong fragment mapping scrambles this.
    M, N, K = 128, 32, 32
    A = torch.arange(M, device="cuda").float().repeat(K, 1).T.contiguous().bfloat16()
    B = torch.eye(N, K, device="cuda", dtype=torch.bfloat16)
    expect = torch.arange(M, device="cuda").float()[:, None].expand(M, N)
    assert torch.equal(m.gemm_pipelined(A, B), expect)


def test_rejects_bad_k():
    A = torch.randn(16, 40, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError) as e:
        m.gemm_pipelined(A, A)
    assert not isinstance(e.value, NotImplementedError)


@pytest.mark.parametrize("stages", [2, 3, 4])
@pytest.mark.parametrize("K", [32, 64, 96, 128, 160, 4096])
def test_k_tiles_fewer_and_more_than_stages(K, stages):
    # KT < STAGES-1 exercises the empty commits in the prologue; KT >> STAGES the steady state
    torch.manual_seed(K)
    A = torch.randn(130, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(200, K, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(m.gemm_pipelined(A, B, stages), A.float() @ B.float().T, rtol=1e-3, atol=1e-2 * (K / 32) ** 0.5)


@pytest.mark.parametrize("stages", [2, 3, 4])
def test_each_k_tile_lands_in_the_right_slot(stages):
    # A[:, k] = k-tile index, B = all ones: C[i, n] = sum_k A[i, k] = 32 * sum(t for t in tiles)
    K = 32 * 9
    A = (torch.arange(K, device="cuda") // 32).float().repeat(64, 1).bfloat16()
    B = torch.ones(16, K, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(m.gemm_pipelined(A, B, stages), torch.full((64, 16), 32.0 * sum(range(9)), device="cuda"))


def test_rejects_bad_stage_count():
    A = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError) as e:
        m.gemm_pipelined(A, A, 5)
    assert not isinstance(e.value, NotImplementedError)
