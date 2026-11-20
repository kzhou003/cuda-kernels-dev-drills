## 怎么用

每道题一小时左右：读讲解页（5–10 分钟）→ 在 `mine/` 里实现 → `./check.sh <编号>` → 卡住 10–15 分钟再按 L1 → L2 → L3 看提示 → 对照参考解和“面试官追问”。

```bash
git clone https://github.com/kzhou003/cuda-kernels-dev-drills && cd cuda-kernels-dev-drills
./check.sh a1                 # 第一次会建 .venv（pytest + numpy），测你的 mine/a1_*.py
./check.sh a1 --solutions     # 测参考解
./check.sh all                # 全部（没有 GPU 时 G 题自动跳过）
```

## 环境

| 题目 | 需要什么 | 说明 |
|---|---|---|
| A、B、C、D（14 道） | Python 3.10+、numpy、pytest | 只用 CPU，整套几秒跑完 |
| G1、G2、G3 | NVIDIA GPU sm_80+、带 CUDA 的 torch、nvcc | **默认在 RTX 5090（sm_120）上测**；A100 / H100 / RTX 30/40 也能跑 |
| G4 | sm_90+（PDL） | RTX 5090、H100、B200 |

GPU 题用 `torch.utils.cpp_extension.load_inline` 编译 `mine/<题>.cu`，第一次 30–60 秒，之后有缓存。用一个装了 CUDA 版 torch 的 Python：

```bash
python3 -m venv --system-site-packages .venv-gpu && .venv-gpu/bin/pip install pytest numpy
export CUDA_HOME=/usr/local/cuda PATH=$CUDA_HOME/bin:$PATH     # nvcc 版本要和 torch 的 CUDA 版本兼容
PYTHON=.venv-gpu/bin/python ./check.sh g1
PYTHON=.venv-gpu/bin/python ./check.sh g2 --bench --solutions   # 和 cuBLAS / SDPA 比速度
```

GPU 题的骨架里有一行 `#define DRILL_STUB`：留着它时测试直接报 `NotImplementedError`，开始写之前删掉。

**5090 上跑不了的**：wgmma、TMA multicast、`setmaxnreg`（需 H100，sm_90a），tcgen05 / TMEM / 2-CTA MMA / Cluster Launch Control（需 B200，sm_100a）。对应的 B3、B4、C1、C3 的 CLC 部分、D3 是 CPU 模型；每道题的讲解页写明了需要哪张卡才能真跑，以及 CUTLASS / FlashAttention 里对应的源码。

## 建议顺序

规则（CPU 题）和代码（GPU 题）交替做：

1. **A1** 算账 → **A2** fragment 布局 → **G1** 5090 上的 mma.sync GEMM
2. **A3** swizzle → **B1** cp.async 流水 → **G2** 多级流水 GEMM
3. **B2** mbarrier → **B3** TMA → **B4** warp specialization（Hopper）
4. **C1** TMEM（Blackwell）→ **C2** block scaling → **C3** tile 调度
5. **D1** online softmax → **D2** FA 分块 → **G3** 5090 上的 FlashAttention-2 → **D3** FA4 的技巧
6. **C4** kernel 之间的时间线 → **G4** 5090 上实测 PDL + CUDA Graph

依赖：G1 用到 A2；G2 用到 B1、G1；G3 用到 A2、D1、D2；G4 用到 C4；B4、C1 用到 B2。

先读知识页 **总览**，再读 **每一代 GPU**；做 D 题前读 **FlashAttention 1 → 4**，做 C4 / G4 前读 **Kernel 之间**。
