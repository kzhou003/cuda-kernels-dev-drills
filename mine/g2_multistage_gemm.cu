// G2 · Multistage cp.async pipeline for the G1 GEMM (Ampere-style; runs on RTX 5090 / sm_120, sm_80+)
//
// Same math, tiles and fragments as G1 (C = A · B^T, CTA 128 x 128 x 32, 8 warps, ldmatrix + mma.sync).
// What changes is ① copy and ④ pipeline of the five GEMM parts:
//   * global -> SMEM uses cp.async.cg (16 B, bypasses registers; src-size 0 = zero-fill for tail rows)
//   * SMEM holds STAGES ring-buffer slots; the prologue issues STAGES-1 tiles
//   * main loop (exactly drill B1):  cp.async.wait_group<STAGES-2>; __syncthreads();
//                                    issue tile k+STAGES-1 into slot (k-1) % STAGES; commit (always);
//                                    compute on slot k % STAGES
// STAGES is a template parameter (2, 3 or 4). 4 stages x (A + B) x 128 x 40 x 2 B = 80 KB of dynamic SMEM, which
// needs cudaFuncSetAttribute (the default cap is 48 KB; sm_120 allows 99 KB per CTA and 100 KB per SM, H100 227 KB).
// Requirements: K % 32 == 0, contiguous tensors. M and N can be anything.

#define DRILL_STUB  // delete this line when you start implementing
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define BM 128
#define BN 128
#define BK 32
#define LDS (BK + 8)          // padded SMEM row, in bf16 elements
#define NTHREADS 256
#define TILE_ELEMS (BM * LDS)       // one A (or B) slot, in bf16 elements

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

// cp.async 16 B global -> shared; when pred is false, src-size 0 makes the hardware write zeros
__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem, bool pred) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  int src_size = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(a), "l"(gmem), "r"(src_size));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// issue the copies of one BM x BK (or BN x BK) tile: 512 vectors of 16 B, 2 per thread
__device__ __forceinline__ void issue_tile(__nv_bfloat16* s, const __nv_bfloat16* g, int rows_left, int K, int tid) {
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int v = tid + i * NTHREADS;
    int r = v / 4, c = (v % 4) * 8;
    bool ok = r < rows_left;
    // the address must stay valid even when the copy is predicated off: clamp to row 0
    cp_async_16(s + r * LDS + c, g + (size_t)(ok ? r : 0) * K + c, ok);
  }
}

// one k16 step of the warp's 64 x 32 sub-tile (identical to G1)
__device__ __forceinline__ void warp_mma_k16(const __nv_bfloat16* sA, const __nv_bfloat16* sB, int kk, int warp_m,
                                             int warp_n, int lane, float acc[4][4][4]) {
  uint32_t a[4][4], b[4][2];
#pragma unroll
  for (int mi = 0; mi < 4; ++mi) {
    int r = warp_m * 64 + mi * 16 + (lane % 8) + 8 * ((lane / 8) % 2);
    ldmatrix_x4(a[mi], sA + r * LDS + kk + 8 * (lane / 16));
  }
#pragma unroll
  for (int nj = 0; nj < 2; ++nj) {
    uint32_t r4[4];
    int r = warp_n * 32 + nj * 16 + (lane % 8) + 8 * (lane / 16);
    ldmatrix_x4(r4, sB + r * LDS + kk + 8 * ((lane / 8) % 2));
    b[2 * nj][0] = r4[0]; b[2 * nj][1] = r4[1];
    b[2 * nj + 1][0] = r4[2]; b[2 * nj + 1][1] = r4[3];
  }
#pragma unroll
  for (int mi = 0; mi < 4; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) mma_16816(acc[mi][ni], a[mi], b[ni]);
}

// ---------------------------------------------------------------- the kernel
template <int STAGES>
__global__ void __launch_bounds__(NTHREADS) gemm_pipelined_kernel(const __nv_bfloat16* __restrict__ A,
                                                            const __nv_bfloat16* __restrict__ B,
                                                            float* __restrict__ C, int M, int N, int K) {
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  __nv_bfloat16* sA = smem;                              // STAGES slots of BM x LDS
  __nv_bfloat16* sB = smem + STAGES * TILE_ELEMS;        // STAGES slots of BN x LDS

  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  const int warp_m = warp / 4;        // 0..1 -> 64 rows each
  const int warp_n = warp % 4;        // 0..3 -> 32 cols each
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

  float acc[4][4][4] = {};            // [mi][ni][c0..c3]

  // TODO(drill): prologue + multistage main loop with cp.async commit / wait_group
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

template <int STAGES>
void launch(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C, int M, int N, int K) {
  const int smem_bytes = 2 * STAGES * TILE_ELEMS * sizeof(__nv_bfloat16);
  static bool attr_set = false;
  if (!attr_set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(gemm_pipelined_kernel<STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_bytes));
    C10_CUDA_CHECK(cudaFuncSetAttribute(gemm_pipelined_kernel<STAGES>, cudaFuncAttributePreferredSharedMemoryCarveout,
                                        cudaSharedmemCarveoutMaxShared));
    attr_set = true;
  }
  if (std::getenv("DRILL_OCC")) {
    int blocks = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, gemm_pipelined_kernel<STAGES>, NTHREADS,
                                                                 smem_bytes));
    printf("STAGES=%d: %d B of SMEM per CTA -> %d CTAs per SM\n", STAGES, smem_bytes, blocks);
  }
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  gemm_pipelined_kernel<STAGES><<<grid, NTHREADS, smem_bytes, c10::cuda::getCurrentCUDAStream()>>>(A, B, C, M, N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// stages = 2, 3 or 4. On an RTX 5090 (100 KB SMEM per SM, 64K registers) 2 stages keep 2 CTAs per SM
// resident while 3-4 stages drop to 1 CTA: compare them with ./check.sh g2 --bench.
torch::Tensor gemm_pipelined(torch::Tensor A, torch::Tensor B, int64_t stages) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.dtype() == torch::kBFloat16 && B.dtype() == torch::kBFloat16);
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && A.dim() == 2 && B.dim() == 2 && A.size(1) == B.size(1));
  int M = A.size(0), N = B.size(0), K = A.size(1);
  TORCH_CHECK(K % BK == 0, "K must be a multiple of 32");
  auto C = torch::empty({M, N}, A.options().dtype(torch::kFloat32));
  auto a = reinterpret_cast<const __nv_bfloat16*>(A.data_ptr());
  auto b = reinterpret_cast<const __nv_bfloat16*>(B.data_ptr());
  switch (stages) {
    case 2: launch<2>(a, b, C.data_ptr<float>(), M, N, K); break;
    case 3: launch<3>(a, b, C.data_ptr<float>(), M, N, K); break;
    case 4: launch<4>(a, b, C.data_ptr<float>(), M, N, K); break;
    default: TORCH_CHECK(false, "stages must be 2, 3 or 4");
  }
  return C;
}
