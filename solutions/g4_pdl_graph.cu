// G4 · Programmatic Dependent Launch (PDL) + CUDA Graph on a chain of small kernels (sm_90+; RTX 5090 = sm_120)
//
// A decode step is a chain of hundreds of tiny kernels. This drill builds such a chain:
//   x_{i+1} = a · x_i + w        (n_kernels launches, ping-ponging between two buffers)
// and gives every kernel a "prologue" that does not depend on the previous kernel's output — it reads its own
// weights and then spins for `spin` cycles to stand in for the real setup work (SMEM carve-up, mbarrier init,
// TMA descriptor prefetch, loading weights / scales).
//
// With PDL:
//   * device side: the kernel calls griddepcontrol.launch_dependents early (here: at entry), so the NEXT kernel's
//     CTAs may be scheduled while this one is still running; and it calls griddepcontrol.wait right before the
//     first access that touches the previous kernel's data. wait blocks until the previous grid has completed and
//     its memory is visible. Everything before wait overlaps with the previous kernel.
//   * host side: launch with cudaLaunchKernelEx + cudaLaunchAttributeProgrammaticStreamSerialization = 1.
//     Without that attribute the griddepcontrol instructions are harmless no-ops and launches serialize as usual.
//   * Both work inside stream capture: the graph keeps "programmatic" edges between the kernel nodes.
//
// Each kernel's block 0 also records %globaltimer (ns) at entry and at exit into `stamps` (if not null), so the
// tests can observe the overlap: with PDL, kernel i+1 starts before kernel i has finished.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

// ---------------------------------------------------------------- given device helpers
__device__ __forceinline__ uint64_t global_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ void spin_cycles(long long n) {
  long long start = clock64();
  while (clock64() - start < n) {
  }
}
__device__ __forceinline__ void pdl_launch_dependents() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
__device__ __forceinline__ void pdl_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}

// ---------------------------------------------------------------- the kernel
// y = a * x + w.  x is the previous kernel's output; y is the buffer the previous kernel READ (ping-pong),
// so writing y early is a write-after-read hazard, and reading x early is a read-after-write hazard.
__global__ void step_kernel(const float* __restrict__ x, float* __restrict__ y, const float* __restrict__ w, int n,
                            float a, long long spin, bool use_pdl, unsigned long long* stamps) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (stamps && blockIdx.x == 0 && threadIdx.x == 0) stamps[0] = global_ns();

  // >>> SOLUTION (where do launch_dependents and wait go?)
  // Let the next kernel get onto the SMs as early as possible. Triggering does NOT publish our results — the
  // dependent's griddepcontrol.wait still waits for this whole grid to finish — so triggering at entry is safe.
  if (use_pdl) pdl_launch_dependents();

  // prologue: touches nothing the previous kernel writes or reads
  const float wi = i < n ? w[i] : 0.f;
  spin_cycles(spin);

  // first access to the previous kernel's output (x) and to its input (y): wait here, not earlier
  if (use_pdl) pdl_wait();
  // <<< SOLUTION

  if (i < n) y[i] = a * x[i] + wi;
  if (stamps && blockIdx.x == 0 && threadIdx.x == 0) stamps[1] = global_ns();
}

// Launch `n_kernels` steps on the current stream: buf0 -> buf1 -> buf0 -> ...; returns the buffer holding the result.
torch::Tensor run_chain(torch::Tensor buf0, torch::Tensor buf1, torch::Tensor w, int64_t n_kernels, double a,
                        int64_t spin, bool use_pdl, c10::optional<torch::Tensor> stamps) {
  TORCH_CHECK(buf0.is_cuda() && buf0.dtype() == torch::kFloat32 && buf0.is_contiguous());
  TORCH_CHECK(buf1.sizes() == buf0.sizes() && w.sizes() == buf0.sizes() && buf1.is_contiguous() && w.is_contiguous());
  const int n = buf0.numel();
  unsigned long long* st = nullptr;
  if (stamps.has_value()) {
    TORCH_CHECK(stamps->dtype() == torch::kInt64 && stamps->numel() >= 2 * n_kernels && stamps->is_contiguous());
    st = reinterpret_cast<unsigned long long*>(stamps->data_ptr<int64_t>());
  }
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  float* bufs[2] = {buf0.data_ptr<float>(), buf1.data_ptr<float>()};
  const float* wp = w.data_ptr<float>();

  // >>> SOLUTION (host: cudaLaunchKernelEx with the programmatic-stream-serialization attribute)
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((n + 255) / 256);
  cfg.blockDim = dim3(256);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = use_pdl ? 1 : 0;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  for (int64_t k = 0; k < n_kernels; ++k) {
    const float* x = bufs[k % 2];
    float* y = bufs[(k + 1) % 2];
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, step_kernel, x, y, wp, n, static_cast<float>(a),
                                      static_cast<long long>(spin), use_pdl, st ? st + 2 * k : nullptr));
  }
  // <<< SOLUTION
  return n_kernels % 2 ? buf1 : buf0;
}
