import pytest

from drill import load
from tests._cuda import cc, requires_cuda, torch

pytestmark = [requires_cuda, pytest.mark.skipif(cc()[0] < 9, reason="PDL needs sm_90+")]
m = load("g4_pdl_graph")


def expected(x0, w, n, a):
    x = x0.double()
    for _ in range(n):
        x = a * x + w.double()
    return x.float()


@pytest.mark.parametrize("use_pdl", [False, True])
@pytest.mark.parametrize("n_kernels", [1, 2, 7, 64])
def test_chain_result(use_pdl, n_kernels):
    torch.manual_seed(n_kernels)
    x0 = torch.randn(100_000, device="cuda")
    w = torch.randn_like(x0)
    b0, b1 = x0.clone(), torch.empty_like(x0)
    out = m.run_chain(b0, b1, w, n_kernels, 0.5, 2000, use_pdl)
    torch.cuda.synchronize()
    assert out.data_ptr() == (b1 if n_kernels % 2 else b0).data_ptr()
    torch.testing.assert_close(out, expected(x0, w, n_kernels, 0.5), rtol=1e-5, atol=1e-5)


def test_without_pdl_kernels_do_not_overlap():
    x = torch.randn(1 << 14, device="cuda")
    st = torch.zeros(2 * 16, dtype=torch.int64, device="cuda")
    m.run_chain(x.clone(), torch.empty_like(x), torch.randn_like(x), 16, 0.5, 20000, False, st)
    torch.cuda.synchronize()
    s = st.view(-1, 2).tolist()
    assert all(s[i + 1][0] >= s[i][1] for i in range(15))


def test_with_pdl_next_kernel_starts_before_previous_ends():
    x = torch.randn(1 << 14, device="cuda")
    st = torch.zeros(2 * 16, dtype=torch.int64, device="cuda")
    m.run_chain(x.clone(), torch.empty_like(x), torch.randn_like(x), 16, 0.5, 20000, True, st)
    torch.cuda.synchronize()
    s = st.view(-1, 2).tolist()
    overlaps = sum(s[i + 1][0] < s[i][1] for i in range(15))
    assert overlaps >= 12, f"only {overlaps}/15 kernels started early"


@pytest.mark.parametrize("use_pdl", [False, True])
def test_works_inside_cuda_graph(use_pdl):
    x0 = torch.randn(50_000, device="cuda")
    w = torch.randn_like(x0)
    b0, b1 = x0.clone(), torch.empty_like(x0)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        m.run_chain(b0, b1, w, 2, 1.0, 0, use_pdl)        # build the extension outside capture
        torch.cuda.synchronize()
        b0.copy_(x0)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            out = m.run_chain(b0, b1, w, 10, 0.5, 1000, use_pdl)
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(3):                                   # each replay applies 10 more steps
        g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected(x0, w, 30, 0.5), rtol=1e-5, atol=1e-5)
