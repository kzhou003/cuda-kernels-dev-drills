"""G3 · FlashAttention-2 forward with mma.sync on RTX 5090 — reference solution (Python wrapper; the work is in the .cu).

Implement the KV loop in `mine/g3_flash_attn.cu` (between the TODO markers), then delete the line
`#define DRILL_STUB`. Until then flash_attn raises NotImplementedError.

- flash_attn(q, k, v, causal=False, scale=None) -> (o, lse)
  q, k, v: (B, H, N, D) bf16 contiguous CUDA tensors with the same shape, D = 64 or 128.
  scale defaults to 1 / sqrt(D). o: (B, H, N, D) bf16 = softmax(scale·q kᵀ [causal]) v.
  lse: (B, H, N) float32, natural-log log-sum-exp of the scaled scores of each row.
  causal masks key j > query i (square attention). N can be any length (tails are masked).

Runs on any sm_80+ GPU; default target RTX 5090. Benchmark vs PyTorch SDPA: ./check.sh g3 --bench
"""
import math

from common.cuda_ext import load_drill

_DECLS = "std::vector<torch::Tensor> flash_attn_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal, double scale);"


def flash_attn(q, k, v, causal=False, scale=None):
    """(o, lse) from the FA2-style kernel in the .cu file next to this wrapper."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    o, lse = load_drill(__file__, _DECLS, ("flash_attn_fwd",)).flash_attn_fwd(q, k, v, causal, scale)
    return o, lse
