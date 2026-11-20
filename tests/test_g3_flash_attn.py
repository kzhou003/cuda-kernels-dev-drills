import math

import pytest

from drill import load
from tests._cuda import requires_cuda, torch

pytestmark = requires_cuda
m = load("g3_flash_attn")


def ref(q, k, v, causal, scale):
    s = (q.float() @ k.float().transpose(-1, -2)) * scale
    if causal:
        N = q.shape[2]
        s = s.masked_fill(torch.ones(N, N, device=q.device, dtype=torch.bool).triu(1), float("-inf"))
    return torch.softmax(s, -1) @ v.float(), torch.logsumexp(s, -1)


def rand(B, H, N, D, seed=0):
    torch.manual_seed(seed)
    return [torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3)]


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("N", [64, 128, 200, 1000])
@pytest.mark.parametrize("causal", [False, True])
def test_matches_reference(D, N, causal):
    q, k, v = rand(2, 3, N, D)
    o, lse = m.flash_attn(q, k, v, causal)
    ro, rl = ref(q, k, v, causal, 1 / math.sqrt(D))
    assert o.dtype == torch.bfloat16 and o.shape == q.shape and lse.shape == (2, 3, N)
    torch.testing.assert_close(o.float(), ro, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, rl, atol=1e-3, rtol=1e-3)


def test_tiny_and_odd_lengths():
    for N in (1, 7, 63, 65):
        q, k, v = rand(1, 1, N, 64, seed=N)
        for causal in (False, True):
            o, lse = m.flash_attn(q, k, v, causal)
            ro, rl = ref(q, k, v, causal, 0.125)
            torch.testing.assert_close(o.float(), ro, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(lse, rl, atol=1e-3, rtol=1e-3)


def test_first_causal_row_copies_first_value():
    q, k, v = rand(1, 2, 128, 64, seed=3)
    o, _ = m.flash_attn(q, k, v, causal=True)
    torch.testing.assert_close(o[:, :, 0].float(), v[:, :, 0].float(), atol=1e-2, rtol=0)


def test_large_scores_do_not_overflow():
    q, k, v = rand(1, 1, 256, 64, seed=4)
    o, lse = m.flash_attn(q * 8, k * 8, v, False)
    ro, rl = ref(q * 8, k * 8, v, False, 0.125)
    assert torch.isfinite(o.float()).all() and torch.isfinite(lse).all()
    torch.testing.assert_close(o.float(), ro, atol=3e-2, rtol=3e-2)


def test_custom_scale():
    q, k, v = rand(1, 2, 96, 128, seed=5)
    o, lse = m.flash_attn(q, k, v, False, scale=0.02)
    ro, rl = ref(q, k, v, False, 0.02)
    torch.testing.assert_close(o.float(), ro, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, rl, atol=1e-3, rtol=1e-3)


def test_rejects_unsupported_head_dim():
    q, k, v = rand(1, 1, 64, 32)
    with pytest.raises(RuntimeError) as e:
        m.flash_attn(q, k, v)
    assert not isinstance(e.value, NotImplementedError)
