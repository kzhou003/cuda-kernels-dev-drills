// G3 · FlashAttention-2 forward with mma.sync on RTX 5090 (sm_120; also sm_80+)
//
// O = softmax(scale · Q Kᵀ [causal mask]) V   for bf16 Q, K, V of shape (B, H, N, D), D ∈ {64, 128}.
// Also returns LSE (natural log, fp32, shape (B, H, N)), like flash_attn's softmax_lse.
//
// One CTA = one (batch, head, 64-query block); 4 warps, each owns 16 query rows (FA2 split: no inter-warp
// reduction needed because every warp sees all keys of its rows).
// Per KV block of 64 keys:
//   S  = Q Kᵀ        : Q fragments live in registers for the whole kernel (loaded once with ldmatrix),
//                      K fragments come from SMEM ([key][d] = N×K layout, non-transposed ldmatrix as in G1)
//   online softmax    : in the log2 domain on the C fragments (each thread owns rows g and g+8; a row is spread
//                      over the 4 threads of a quad -> reduce with __shfl_xor_sync(…, 1) and (…, 2))
//   O += P V          : P is converted to bf16 and re-used directly as the A operand — the C layout of S and the
//                      A layout of the next mma line up row by row (drill A2) — and V fragments come from SMEM
//                      with ldmatrix.trans, because V is stored [key][d] (K×N, n contiguous).
// Causal (square, top-left == bottom-right): KV blocks after the diagonal are never loaded; only the diagonal
// block applies an element mask. Keys >= N (tail) are masked; query rows >= N are not written.
// Loads are synchronous (uint4 + __syncthreads): the drill is the fragment-level math, not the pipeline (G2).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cstdint>

#define BR 64
#define BC 64
#define NWARPS 4

// ---------------------------------------------------------------- given helpers
__device__ __forceinline__ void ldmatrix_x4(uint32_t r[4], const void* smem) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t r[4], const void* smem) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma_16816(float d[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// two fp32 -> one b32 holding two bf16 (low half = first argument)
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
// rows x D tile from global (row stride D) into SMEM (row stride D + 8); rows >= rows_left become zeros
template <int D>
__device__ __forceinline__ void load_rows(__nv_bfloat16* s, const __nv_bfloat16* g, int rows, int rows_left, int tid) {
  constexpr int VPR = D / 8;               // 16-byte vectors per row
  for (int v = tid; v < rows * VPR; v += NWARPS * 32) {
    int r = v / VPR, c = (v % VPR) * 8;
    uint4 val = make_uint4(0, 0, 0, 0);
    if (r < rows_left) val = *reinterpret_cast<const uint4*>(g + (size_t)r * D + c);
    *reinterpret_cast<uint4*>(s + r * (D + 8) + c) = val;
  }
}

// ---------------------------------------------------------------- the kernel
template <int D>
__global__ void __launch_bounds__(NWARPS * 32) flash_fwd_kernel(const __nv_bfloat16* __restrict__ Q,
                                                                const __nv_bfloat16* __restrict__ K,
                                                                const __nv_bfloat16* __restrict__ V,
                                                                __nv_bfloat16* __restrict__ O, float* __restrict__ LSE,
                                                                int N, float scale_log2, bool causal) {
  constexpr int LD = D + 8;                // padded SMEM row (bank-conflict-free ldmatrix, see A3)
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  __nv_bfloat16* sQ = smem;                // BR x LD
  __nv_bfloat16* sK = sQ + BR * LD;        // BC x LD
  __nv_bfloat16* sV = sK + BC * LD;        // BC x LD

  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  const int g = lane / 4, t = lane % 4;
  const int q0 = blockIdx.x * BR;
  const size_t bh = (size_t)blockIdx.y * N * D;          // (batch, head) offset
  const __nv_bfloat16* Qg = Q + bh;
  const __nv_bfloat16* Kg = K + bh;
  const __nv_bfloat16* Vg = V + bh;

  // Q tile -> SMEM -> registers (A fragments for every k16 step of d), kept for the whole kernel
  load_rows<D>(sQ, Qg + (size_t)q0 * D, BR, N - q0, tid);
  __syncthreads();
  uint32_t qf[D / 16][4];
#pragma unroll
  for (int kd = 0; kd < D / 16; ++kd) {
    int r = warp * 16 + (lane % 8) + 8 * ((lane / 8) % 2);
    ldmatrix_x4(qf[kd], sQ + r * LD + kd * 16 + 8 * (lane / 16));
  }

  float o[D / 8][4] = {};                  // O accumulator: 16 rows x D per warp, C layout
  float m[2] = {-INFINITY, -INFINITY};     // running max (log2 domain) of rows g and g+8
  float l[2] = {0.f, 0.f};                 // running sum, per thread (partial over its columns)

  // >>> SOLUTION (KV loop: S = QKᵀ, mask, online softmax, O += P V; then the epilogue normalisation)
  const int n_blocks = causal ? min((N + BC - 1) / BC, (q0 + BR + BC - 1) / BC) : (N + BC - 1) / BC;
  for (int jb = 0; jb < n_blocks; ++jb) {
    const int k0 = jb * BC;
    __syncthreads();                       // previous iteration is done reading sK / sV
    load_rows<D>(sK, Kg + (size_t)k0 * D, BC, N - k0, tid);
    load_rows<D>(sV, Vg + (size_t)k0 * D, BC, N - k0, tid);
    __syncthreads();

    // ---- S = Q Kᵀ : 16 x 64 per warp = 8 n8 tiles
    float s[BC / 8][4] = {};
#pragma unroll
    for (int kd = 0; kd < D / 16; ++kd) {
#pragma unroll
      for (int nj = 0; nj < BC / 16; ++nj) {
        uint32_t r4[4];
        int r = nj * 16 + (lane % 8) + 8 * (lane / 16);
        ldmatrix_x4(r4, sK + r * LD + kd * 16 + 8 * ((lane / 8) % 2));
        uint32_t b0[2] = {r4[0], r4[1]}, b1[2] = {r4[2], r4[3]};
        mma_16816(s[2 * nj], qf[kd], b0);
        mma_16816(s[2 * nj + 1], qf[kd], b1);
      }
    }

    // ---- scale into the log2 domain and mask (tail keys always, causal only on the diagonal block)
    const bool diag = causal && (k0 + BC > q0);
#pragma unroll
    for (int ni = 0; ni < BC / 8; ++ni)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        int key = k0 + ni * 8 + 2 * t + (e & 1);
        int qry = q0 + warp * 16 + g + 8 * (e >> 1);
        bool dead = key >= N || (diag && key > qry);
        s[ni][e] = dead ? -INFINITY : s[ni][e] * scale_log2;
      }

    // ---- online softmax per row (h = 0: row g, h = 1: row g + 8)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float mx = m[h];
#pragma unroll
      for (int ni = 0; ni < BC / 8; ++ni) mx = fmaxf(mx, fmaxf(s[ni][2 * h], s[ni][2 * h + 1]));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));   // the 4 threads of a quad share a row
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
      // a row that has seen only masked keys keeps m = -inf: use 0 as the base so exp2 never sees inf - inf
      const float base = mx == -INFINITY ? 0.f : mx;
      const float alpha = exp2f(m[h] - base);               // m[h] = -inf on the first block -> alpha = 0
      m[h] = mx;
      float rs = 0.f;
#pragma unroll
      for (int ni = 0; ni < BC / 8; ++ni) {
        s[ni][2 * h] = exp2f(s[ni][2 * h] - base);
        s[ni][2 * h + 1] = exp2f(s[ni][2 * h + 1] - base);
        rs += s[ni][2 * h] + s[ni][2 * h + 1];
      }
      l[h] = l[h] * alpha + rs;                               // per-thread partial; quad-reduced at the end
#pragma unroll
      for (int nd = 0; nd < D / 8; ++nd) {
        o[nd][2 * h] *= alpha;
        o[nd][2 * h + 1] *= alpha;
      }
    }

    // ---- O += P V : P (C layout, fp32) -> A fragments (bf16) without any shuffle
#pragma unroll
    for (int kk = 0; kk < BC / 16; ++kk) {
      uint32_t pa[4] = {pack_bf16(s[2 * kk][0], s[2 * kk][1]),           // row g,   keys 16kk + 2t, +1
                        pack_bf16(s[2 * kk][2], s[2 * kk][3]),           // row g+8
                        pack_bf16(s[2 * kk + 1][0], s[2 * kk + 1][1]),   // row g,   keys 16kk + 8 + 2t, +1
                        pack_bf16(s[2 * kk + 1][2], s[2 * kk + 1][3])};  // row g+8
#pragma unroll
      for (int nd = 0; nd < D / 16; ++nd) {
        // V is [key][d]: transposed ldmatrix gives the B layout (pairs of consecutive keys per register).
        // Matrices: (keys 0-7, d 0-7), (keys 8-15, d 0-7), (keys 0-7, d 8-15), (keys 8-15, d 8-15).
        uint32_t r4[4];
        int r = kk * 16 + (lane % 8) + 8 * ((lane / 8) % 2);
        int c = nd * 16 + 8 * (lane / 16);
        ldmatrix_x4_trans(r4, sV + r * LD + c);
        uint32_t b0[2] = {r4[0], r4[1]}, b1[2] = {r4[2], r4[3]};
        mma_16816(o[2 * nd], pa, b0);
        mma_16816(o[2 * nd + 1], pa, b1);
      }
    }
  }

  // ---- epilogue: finish the row sums across the quad, normalise, write O and LSE
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    l[h] += __shfl_xor_sync(0xffffffff, l[h], 1);
    l[h] += __shfl_xor_sync(0xffffffff, l[h], 2);
  }
  // <<< SOLUTION

#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int row = q0 + warp * 16 + g + 8 * h;
    const float inv = l[h] > 0.f ? 1.f / l[h] : 0.f;
    if (row < N) {
#pragma unroll
      for (int nd = 0; nd < D / 8; ++nd) {
        int col = nd * 8 + 2 * t;
        *reinterpret_cast<__nv_bfloat162*>(O + bh + (size_t)row * D + col) =
            __floats2bfloat162_rn(o[nd][2 * h] * inv, o[nd][2 * h + 1] * inv);
      }
      // m is in the log2 domain: lse = (m + log2 l) · ln 2
      if (t == 0) LSE[(size_t)blockIdx.y * N + row] = l[h] > 0.f ? (m[h] + log2f(l[h])) * 0.69314718056f : -INFINITY;
    }
  }
}

template <int D>
void launch(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, torch::Tensor& o,
            torch::Tensor& lse, int N, float scale_log2, bool causal) {
  const int smem = (BR + 2 * BC) * (D + 8) * sizeof(__nv_bfloat16);
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(flash_fwd_kernel<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    attr = true;
  }
  dim3 grid((N + BR - 1) / BR, q.size(0) * q.size(1));
  flash_fwd_kernel<D><<<grid, NWARPS * 32, smem, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()), reinterpret_cast<__nv_bfloat16*>(o.data_ptr()),
      lse.data_ptr<float>(), N, scale_log2, causal);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> flash_attn_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal,
                                          double scale) {
  TORCH_CHECK(q.is_cuda() && q.dtype() == torch::kBFloat16 && k.dtype() == q.dtype() && v.dtype() == q.dtype());
  TORCH_CHECK(q.dim() == 4 && q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q, k, v must all be (B, H, N, D)");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous());
  const int N = q.size(2), D = q.size(3);
  auto o = torch::empty_like(q);
  auto lse = torch::empty({q.size(0), q.size(1), N}, q.options().dtype(torch::kFloat32));
  const float scale_log2 = static_cast<float>(scale) * 1.4426950408889634f;
  if (D == 64) launch<64>(q, k, v, o, lse, N, scale_log2, causal);
  else if (D == 128) launch<128>(q, k, v, o, lse, N, scale_log2, causal);
  else TORCH_CHECK(false, "head dim must be 64 or 128");
  return {o, lse};
}
