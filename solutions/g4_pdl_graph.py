"""G4 · PDL + CUDA Graph on a chain of small kernels (RTX 5090) — reference solution (Python wrapper; the work is in the .cu).

Implement the two TODO regions in `mine/g4_pdl_graph.cu` (device: where launch_dependents / wait go; host: the
cudaLaunchKernelEx launch with the PDL attribute), then delete the line `#define DRILL_STUB`. Until then
run_chain raises NotImplementedError.

- run_chain(buf0, buf1, w, n_kernels, a, spin, use_pdl, stamps=None) -> tensor
  Launches n_kernels kernels on the current stream. Kernel k reads buffer k % 2 and writes buffer (k+1) % 2:
  y = a·x + w (float32, all three tensors contiguous, same shape). Returns the buffer holding the final result
  (buf1 if n_kernels is odd, else buf0). Each kernel spins `spin` clock cycles in a prologue that does not touch
  the buffers. use_pdl=True launches with programmatic stream serialization and the kernels use
  griddepcontrol.launch_dependents / griddepcontrol.wait; results must be identical either way.
  stamps (int64 CUDA tensor with ≥ 2·n_kernels elements, optional): kernel k's block 0 writes its %globaltimer
  (ns) at entry to stamps[2k] and at exit to stamps[2k+1].
  Works under torch.cuda.graph capture.

Needs sm_90+ (PDL); default target RTX 5090 (sm_120). Benchmark stream / graph / +PDL: ./check.sh g4 --bench
"""
from common.cuda_ext import load_drill

_DECLS = ("torch::Tensor run_chain(torch::Tensor buf0, torch::Tensor buf1, torch::Tensor w, int64_t n_kernels, "
          "double a, int64_t spin, bool use_pdl, c10::optional<torch::Tensor> stamps);")


def run_chain(buf0, buf1, w, n_kernels, a, spin, use_pdl, stamps=None):
    """Launch the kernel chain (see module docstring)."""
    return load_drill(__file__, _DECLS, ("run_chain",)).run_chain(buf0, buf1, w, n_kernels, a, spin, use_pdl, stamps)
