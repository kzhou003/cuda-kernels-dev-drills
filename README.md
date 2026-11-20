# cuda-kernels-dev-drills

18 one-hour drills on how CUDA kernels change across NVIDIA GPU generations, built around GEMM and FlashAttention.
14 CPU drills rebuild one hardware mechanism each in Python (mma.sync fragment layouts, SMEM swizzle, cp.async
pipelines, mbarrier phases, TMA boxes and multicast, warp specialization, TMEM accumulators, MXFP8/NVFP4 block
scaling, Stream-K, PDL timelines, FlashAttention 1→4 math), mirroring CUTLASS 4.9, FlashAttention, DeepGEMM and
SGLang at pinned commits. 4 GPU drills are real CUDA kernels tested on an RTX 5090 (sm_120): an ldmatrix + mma.sync
BF16 GEMM, a multistage cp.async GEMM (~0.9× cuBLAS), a FlashAttention-2 forward pass, and a PDL + CUDA Graph chain
that cuts per-kernel time from 4.2 µs to 0.7 µs. Quick start: `./check.sh a1`. The rest of this README is in Chinese.

---

18 道一小时的练习，讲清楚每一代 NVIDIA GPU（Volta → Ampere → Hopper → Blackwell，以及 RTX 5090）上 kernel 代码为什么要那样写，主线是 GEMM 和 FlashAttention：

- **14 道 CPU 题**（A–D）：每题用 Python 复刻一个硬件机制的“规则”，对照 CUTLASS 4.9、FlashAttention、DeepGEMM、SGLang 的固定版本源码。不需要 GPU。
- **4 道 GPU 题**（G）：在 **RTX 5090（sm_120）** 上写真实 CUDA kernel，并和 cuBLAS / PyTorch SDPA 比速度。
- 讲解页有图、类比、逐步演示、三级提示和带注释的参考解；“PDL 为什么只在 graph / 编译器层面才有用”有专门的知识页和两道题（C4 算、G4 测）。

**在线阅读：https://kzhou003.github.io/cuda-kernels-dev-drills/** · [PDF（全部提示和参考解展开）](cuda-kernels-dev-drills.pdf) · 本地打开 `index.html` 也可以。

- `mine/` 接口骨架（你写的地方）· `tests/` 验收测试 · `solutions/` 带注释的参考解
- 每题的讲解与提示：`hints/<stem>.html`；领域知识：`knowledge/`（总览、每一代 GPU、FlashAttention 1→4、Kernel 之间）

## 快速开始

```bash
./check.sh a1                 # 第一次会建 .venv（pytest + numpy），测你写的 mine/a1_*.py
./check.sh a1 --solutions     # 测参考解
./check.sh all                # 全部；没有 GPU 时 G 题自动跳过
```

GPU 题需要一个装了 CUDA 版 torch 的 Python 和 nvcc：

```bash
export CUDA_HOME=/usr/local/cuda PATH=$CUDA_HOME/bin:$PATH
PYTHON=/path/to/python-with-torch ./check.sh g1
PYTHON=/path/to/python-with-torch ./check.sh g2 --bench --solutions   # 和 cuBLAS 比
```

| 题目 | 在哪跑 | 说明 |
|---|---|---|
| A、B、C、D | 任何机器（CPU） | 整套几秒 |
| G1、G2、G3 | NVIDIA sm_80+，**默认 RTX 5090** | A100 / H100 / RTX 30/40 也行 |
| G4 | sm_90+（PDL） | RTX 5090、H100、B200 |
| B3 的 multicast、B4、C1、D3 的真实版本 | 需要 H100（sm_90a）或 B200（sm_100a） | 5090 没有 wgmma / TMEM / tcgen05 / multicast，所以这几道是 CPU 模型，讲解页写明了对应的 CUTLASS / FA 源码 |

## 怎么做一道题

1. 读讲解页（5–10 分钟）：题面速览 → 类比 → 图解 → 逐步演示 → 易错点。
2. 在 `mine/` 里实现，限时为卡片上的分钟数。GPU 题先删掉 `.cu` 里的 `#define DRILL_STUB`。
3. 跑测试；卡住 10–15 分钟再按 L1 → L2 → L3 看提示。
4. 对照参考解和“面试官追问 & 对照源码”。

建议顺序（规则和代码交替）：A1 → A2 → G1 → A3 → B1 → G2 → B2 → B3 → B4 → C1 → C2 → C3 → D1 → D2 → G3 → D3 → C4 → G4。

## 进度表

| ✓ | 编号 | 题目 | 分钟 | 难度 |
|---|---|---|---|---|
| ☐ | A1 | Roofline、tile 算术强度与寄存器 / SMEM 预算 | 35 | 易 |
| ☐ | A2 | mma.sync m16n8k16 的 fragment 布局与 ldmatrix | 60 | 中 |
| ☐ | A3 | SMEM bank conflict 与 CuTe Swizzle<B,M,S> | 45 | 中 |
| ☐ | B1 | cp.async 多级流水：commit / wait_group 与 K 循环时间线 | 50 | 中 |
| ☐ | B2 | mbarrier：相位位、事务字节与 TMA 生产者/消费者环 | 60 | 难 |
| ☐ | B3 | TMA box 拷贝：越界填零、128B swizzle、cluster multicast | 50 | 中 |
| ☐ | B4 | Warp specialization：Pingpong vs Cooperative 与 setmaxnreg | 45 | 中 |
| ☐ | C1 | Blackwell TMEM：列分配、双缓冲累加器与 2-CTA MMA | 45 | 中 |
| ☐ | C2 | MXFP8 / NVFP4 block scaling 与 DeepGEMM 的 FP8 提升 | 60 | 难 |
| ☐ | C3 | Tile 调度：wave quantization、raster swizzle、persistent、Stream-K | 55 | 中 |
| ☐ | C4 | Kernel 之间：stream vs CUDA Graph vs PDL 的时间线与依赖 | 50 | 中 |
| ☐ | D1 | Online softmax、(O, LSE) 状态与 split-KV 合并 | 45 | 中 |
| ☐ | D2 | FlashAttention 前向分块：循环顺序、causal 跳块与 HBM 流量 | 55 | 中 |
| ☐ | D3 | FA4 的 softmax：FMA 上的 exp2 多项式与条件 rescale | 50 | 中 |
| ☐ | G1 | 5090：ldmatrix + mma.sync 的 BF16 GEMM | 75 | 难 |
| ☐ | G2 | 5090：cp.async 多级流水 GEMM（2 / 3 / 4 级） | 60 | 难 |
| ☐ | G3 | 5090：FlashAttention-2 前向（mma.sync + online softmax） | 90 | 难 |
| ☐ | G4 | 5090：PDL + CUDA Graph，测出 kernel 之间省下的时间 | 60 | 中 |

## RTX 5090 上的参考数字

参考解在 RTX 5090（driver 580、CUDA 13.0、torch 2.13）上的结果：

- **G1** 同步 mma.sync GEMM：8192³ 182.9 TF/s，cuBLAS 的 0.87×
- **G2** 2 级 cp.async：8192³ 189.1 TF/s（0.90×）；3、4 级反而只有 0.81×，因为 SMEM 超过 50 KB 后每 SM 只能放 1 个 CTA
- **G3** FlashAttention-2 前向：PyTorch SDPA 的 0.54–0.86×（K/V 是同步加载，留作扩展）
- **G4** 64 个小 kernel：stream 266.5 µs → stream + PDL 233.9 → CUDA graph 143.8 → graph + PDL 43.6

## 重新生成页面

页面由 practice-drills-zh 工具链（`build_drills.py`）生成：

```bash
bash <toolkit>/run.sh build_drills.py drills.json --strict --pdf
python3 tools/make_cu_stubs.py --check     # mine/*.cu 骨架检查
```

固定版本：CUTLASS `af95104`（4.9.0）、flash-attention `dbabcc7`、DeepGEMM `5081f85`、SGLang `97cd38e`（见 `drills.json → pins`）。

## License

原创内容（讲解、练习、参考解）MIT。链接到的上游代码遵循其各自的许可证。
