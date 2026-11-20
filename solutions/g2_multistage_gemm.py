"""G2 · Multistage cp.async pipeline GEMM on RTX 5090 — reference solution (Python wrapper; the work is in the .cu).

Implement the prologue and main loop in `mine/g2_multistage_gemm.cu` (between the TODO markers), then delete the
line `#define DRILL_STUB`. Until then gemm_pipelined raises NotImplementedError.

- gemm_pipelined(A, B, stages=2): A (M, K) bf16, B (N, K) bf16, contiguous CUDA tensors, K % 32 == 0.
  Returns C = A @ B.T as (M, N) float32; M and N may be any size. stages must be 2, 3 or 4
  (anything else raises RuntimeError).

Runs on any sm_80+ GPU (needs 80 KB of dynamic SMEM per CTA); default target RTX 5090. Benchmark: ./check.sh g2 --bench
"""
from common.cuda_ext import load_drill

_DECLS = "torch::Tensor gemm_pipelined(torch::Tensor A, torch::Tensor B, int64_t stages);"


def gemm_pipelined(A, B, stages=2):
    """C = A @ B.T in fp32 via the multistage kernel (stages = 2, 3 or 4) in the .cu file next to this wrapper."""
    return load_drill(__file__, _DECLS, ("gemm_pipelined",)).gemm_pipelined(A, B, stages)
