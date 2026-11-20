"""G1 · BF16 GEMM with ldmatrix + mma.sync on RTX 5090 — reference solution (Python wrapper; the work is in the .cu).

Implement the main loop in `mine/g1_mma_gemm.cu` (between the TODO markers), then delete the line
`#define DRILL_STUB`. Until then gemm_mma raises NotImplementedError.

- gemm_mma(A, B): A (M, K) bf16, B (N, K) bf16, both contiguous CUDA tensors, K % 32 == 0.
  Returns C = A @ B.T as (M, N) float32. M and N may be any size (tails are zero-filled / masked).

Runs on any sm_80+ GPU; the default target is the RTX 5090 (sm_120). Benchmark: ./check.sh g1 --bench
"""
from common.cuda_ext import load_drill

_DECLS = "torch::Tensor gemm_mma(torch::Tensor A, torch::Tensor B);"


def gemm_mma(A, B):
    """C = A @ B.T in fp32 via the Tensor Core kernel in the .cu file next to this wrapper."""
    return load_drill(__file__, _DECLS, ("gemm_mma",)).gemm_mma(A, B)
