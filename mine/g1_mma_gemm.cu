// G1 · BF16 GEMM on Tensor Cores with ldmatrix + mma.sync.m16n8k16 (RTX 5090 / sm_120; also sm_80+)
//
// C[M,N] (fp32) = A[M,K] (bf16, row-major) · B[N,K]^T (bf16, row-major = "TN" GEMM)
//
// CTA tile 128 x 128 x 32, 256 threads = 8 warps arranged 2 (M) x 4 (N); each warp owns 64 x 32 of C
// = 4 (m16) x 4 (n8) mma tiles = 64 fp32 accumulators per thread.
// Main loop per K tile (synchronous, no pipelining — that is G2):
//   1. every thread copies 2 x 16 B of A and 2 x 16 B of B from global into SMEM (rows >= M / N -> zeros)
//   2. __syncthreads()
//   3. for each k16 step: ldmatrix.x4 the A fragments (4 per warp), ldmatrix.x4 the B fragments
//      (2 n8 tiles per instruction), then 16 mma.sync
//   4. __syncthreads() before the next tile overwrites SMEM
// SMEM rows are padded from 32 to 40 bf16 (80 B) so that ldmatrix's 8 row addresses fall in 8 different
// 16-byte bank groups (A3: (5r + c) % 8 is a permutation because 5 is odd).
// Requirements: K % 32 == 0, tensors contiguous on the GPU. M and N can be anything.

#define DRILL_STUB  // delete this line when you start implementing
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cstdint>

#define BM 128
#define BN 128
#define BK 32
#define LDS (BK + 8)          // padded SMEM row, in bf16 elements
#define NTHREADS 256

// ---------------------------------------------------------------- given PTX wrappers
__device__ __forceinline__ void ldmatrix_x4(uint32_t r[4], const void* smem) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

// d += a * b   (a: 4 x b32 = 8 bf16, b: 2 x b32 = 4 bf16, d: 4 fp32) — layouts as in drill A2
__device__ __forceinline__ void mma_16816(float d[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// copy one BM x BK (or BN x BK) tile: 128 rows x 4 vectors of 8 bf16 = 512 vectors, 2 per thread
__device__ __forceinline__ void load_tile(__nv_bfloat16* s, const __nv_bfloat16* g, int rows_left, int K, int tid) {
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int v = tid + i * NTHREADS;
    int r = v / 4, c = (v % 4) * 8;
    uint4 val = make_uint4(0, 0, 0, 0);
    if (r < rows_left) val = *reinterpret_cast<const uint4*>(g + (size_t)r * K + c);
    *reinterpret_cast<uint4*>(s + r * LDS + c) = val;
  }
}

// ---------------------------------------------------------------- the kernel
__global__ void __launch_bounds__(NTHREADS) gemm_mma_kernel(const __nv_bfloat16* __restrict__ A,
                                                            const __nv_bfloat16* __restrict__ B,
                                                            float* __restrict__ C, int M, int N, int K) {
  __shared__ __align__(16) __nv_bfloat16 sA[BM * LDS];
  __shared__ __align__(16) __nv_bfloat16 sB[BN * LDS];

  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  const int warp_m = warp / 4;        // 0..1 -> 64 rows each
  const int warp_n = warp % 4;        // 0..3 -> 32 cols each
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

  float acc[4][4][4] = {};            // [mi][ni][c0..c3]

  // TODO(drill): main loop: global -> SMEM, ldmatrix, mma.sync
  // (see the hint page; the reference is in solutions/)


  // epilogue: C fragment layout (A2): c0,c1 -> row g, cols 2t, 2t+1; c2,c3 -> row g+8
  const int g = lane / 4, t = lane % 4;
#pragma unroll
  for (int mi = 0; mi < 4; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      int row = m0 + warp_m * 64 + mi * 16 + g;
      int col = n0 + warp_n * 32 + ni * 8 + 2 * t;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        int rr = row + 8 * h;
        if (rr < M && col < N) C[(size_t)rr * N + col] = acc[mi][ni][2 * h];
        if (rr < M && col + 1 < N) C[(size_t)rr * N + col + 1] = acc[mi][ni][2 * h + 1];
      }
    }
}

torch::Tensor gemm_mma(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.dtype() == torch::kBFloat16 && B.dtype() == torch::kBFloat16);
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && A.dim() == 2 && B.dim() == 2 && A.size(1) == B.size(1));
  int M = A.size(0), N = B.size(0), K = A.size(1);
  TORCH_CHECK(K % BK == 0, "K must be a multiple of 32");
  auto C = torch::empty({M, N}, A.options().dtype(torch::kFloat32));
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  gemm_mma_kernel<<<grid, NTHREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(B.data_ptr()),
      C.data_ptr<float>(), M, N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return C;
}
