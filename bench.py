"""Benchmarks for the G track (needs an NVIDIA GPU; numbers in the docs are from an RTX 5090).

    python bench.py g1            # your mine/ implementation
    DRILL_IMPL=solutions python bench.py g2
    ./check.sh g3 --bench         # same thing through check.sh
"""
import sys

import torch

from drill import load


def timeit(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3     # microseconds


def bench_gemm(name, fn_name, variants=((), )):
    m = load(name)
    fn = getattr(m, fn_name)
    for extra in variants:
        if extra:
            print(f"--- stages = {extra[0]}")
        print(f"{'M=N=K':>8} {'drill TF/s':>11} {'cuBLAS TF/s':>12} {'ratio':>6}")
        for n in (1024, 2048, 4096, 8192):
            A = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
            B = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
            t = timeit(lambda: fn(A, B, *extra))
            tr = timeit(lambda: torch.matmul(A, B.T).float())   # same work: bf16 GEMM with an fp32 result
            fl = 2 * n ** 3
            print(f"{n:>8} {fl / t / 1e6:>11.1f} {fl / tr / 1e6:>12.1f} {tr / t:>6.2f}")


def bench_attn():
    m = load("g3_flash_attn")
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    print(f"{'N':>6} {'d':>4} {'causal':>6} {'drill TF/s':>11} {'SDPA TF/s':>10} {'ratio':>6}")
    for d in (64, 128):
        for N in (1024, 4096, 8192):
            for causal in (False, True):
                q, k, v = (torch.randn(4, 16, N, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
                t = timeit(lambda: m.flash_attn(q, k, v, causal), iters=20)
                tr = timeit(lambda: sdpa(q, k, v, is_causal=causal), iters=20)
                fl = 4 * 4 * 16 * N * N * d / (2 if causal else 1)
                print(f"{N:>6} {d:>4} {str(causal):>6} {fl / t / 1e6:>11.1f} {fl / tr / 1e6:>10.1f} {tr / t:>6.2f}")


def bench_pdl():
    m = load("g4_pdl_graph")
    n_kernels, spin = 64, 4000
    x = torch.randn(1 << 16, device="cuda")
    w = torch.randn_like(x)
    a, b = x.clone(), torch.empty_like(x)
    rows = []
    for pdl in (False, True):
        rows.append((f"stream{' + PDL' if pdl else ''}", timeit(lambda: m.run_chain(a, b, w, n_kernels, 0.5, spin, pdl), 20)))
    for pdl in (False, True):
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            m.run_chain(a, b, w, n_kernels, 0.5, spin, pdl)      # warm up (build) outside capture
            torch.cuda.synchronize()
            with torch.cuda.graph(g, stream=s):
                m.run_chain(a, b, w, n_kernels, 0.5, spin, pdl)
        torch.cuda.current_stream().wait_stream(s)
        rows.append((f"CUDA graph{' + PDL' if pdl else ''}", timeit(g.replay, 20)))
    print(f"{n_kernels} small kernels, prologue spin {spin} cycles")
    for name, t in rows:
        print(f"{name:>20}: {t:8.1f} us total, {t / n_kernels:6.2f} us per kernel")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which.startswith("g1"):
        bench_gemm("g1_mma_gemm", "gemm_mma")
    elif which.startswith("g2"):
        bench_gemm("g2_multistage_gemm", "gemm_pipelined", ((2,), (3,), (4,)))
    elif which.startswith("g3"):
        bench_attn()
    elif which.startswith("g4"):
        bench_pdl()
    else:
        sys.exit("usage: python bench.py g1|g2|g3|g4  (only the G track has benchmarks)")
