lede: 从 Volta 起矩阵乘是一条 warp 指令，但 32 个线程各自只拿着矩阵的几个元素。写 Ampere 风格 kernel 的核心，就是让每个线程寄存器里放的恰好是 mma 要的那几个——ldmatrix 让这件事只要一条指令。
# A2 · mma.sync m16n8k16 的 fragment 布局与 ldmatrix

## 题面速览

- **在系统中的位置**：CuTe 把一条 PTX `mma` 包成一个 “MMA atom”，例如 [[cutlass:include/cute/arch/mma_sm80.hpp@SM80_16x8x16_F32BF16BF16F32_TN|SM80_16x8x16_F32BF16BF16F32_TN]]（A 4 个寄存器、B 2 个、C 4 个），线程到元素的映射写在 [[cutlass:include/cute/atom/mma_traits_sm80.hpp|mma_traits_sm80.hpp]] 的 `ALayout / BLayout / CLayout` 里。`ldmatrix` 是 [[cutlass:include/cute/arch/copy_sm75.hpp@SM75_U32x4_LDSM_N|SM75_U32x4_LDSM_N]]（Turing 引入）。RTX 5090 上 G1、G3 用的就是这两条指令。
- **你要实现**：`a_coords / b_coords / c_coords`（每个 lane 持有哪些元素）、`mma_m16n8k16`（warp 级 MMA 的语义）、`ldmatrix`（谁给地址、谁拿数据）、`a_tile_row_addrs / b_tile_row_addrs`（让 ldmatrix 直接读出 fragment 布局的地址）。
- **难点在这几条规则**：
  1. `g = lane / 4`（行组）、`t = lane % 4`（组内序号）；**一个 quad（4 个相邻 lane）共同拥有两整行**（g 和 g+8）。
  2. 每个 32-bit 寄存器装**两个相邻**元素：A、C 是同一行相邻两列；B 是同一列相邻两个 k。
  3. A 的寄存器顺序是 **左上、左下、右上、右下** 四个 8×8 子块。
  4. ldmatrix：lane `8j..8j+7` 给出第 j 个 8×8 矩阵的 8 行地址；lane l 拿到每个矩阵第 `l/4` 行的第 `2(l%4)`、`+1` 列。

**验收要点**：lane 0 / 5 的坐标 · 每个元素恰好属于一个 lane · warp MMA = numpy 矩阵乘 · C 布局和 A 布局按行对齐 · ldmatrix.x1 的分发 · num 非法 · x4 + A 地址 = A fragment（多种 ld）· x2 + B 地址 = B fragment · 从“SMEM”到 MMA 的完整链路。

## 直觉与类比

::: callout analogy 32 个人一起拼一幅 16×16 的拼图
一个 warp 是 32 个人围着桌子拼图。规则是：每 4 个人一组（quad），第 g 组负责第 g 行和第 g+8 行；组里第 t 个人负责这两行的第 2t、2t+1 列和第 2t+8、2t+9 列——每次两块相邻的拼一起放进一个口袋（寄存器）。`ldmatrix` 像是一个分拣员：32 个人各报一个“这一行在哪个抽屉里”的地址（8 个人报一个 8×8 小块的 8 行），分拣员把整块搬出来，再按“每 4 人一行、每人两块”的规则塞进各自口袋。报地址的人和拿到拼图的人，**不是同一套对应关系**——这是最容易绕晕的地方。
:::

## 图解：A 的 4 个寄存器 = 4 个 8×8 子块

::: flow 16×16 的 A tile 在 lane l 手里的样子
cfg cw=200 rh=100 nw=172
n a0 0 0 "a0：左上" "行 g，列 2t,2t+1" k=acc
n a2 1 0 "a2：右上" "行 g，列 2t+8,+9" k=vio
n a1 0 1 "a1：左下" "行 g+8，列 2t,2t+1" k=acc
n a3 1 1 "a3：右下" "行 g+8，列 2t+8,+9" k=vio
n ld 2.3 0 "ldmatrix.x4" "矩阵 j ← lane 8j..8j+7" k=teal
n mma 2.3 1 "mma.sync m16n8k16" "D = A·B + C" k=warn
e ld -> a2
e ld:b -> mma:t
e a3 -> mma
cap: x4 的四个矩阵依次是左上、左下、右上、右下，正好对应 a0..a3：`a_tile_row_addrs` 里矩阵 j 的行 = `r + 8·(j % 2)`，列 = `8·(j / 2)`。C 的 c0..c3 和 a0、a1 用同一套“行 g / 行 g+8、列 2t”规律，所以 S = QKᵀ 的结果可以不经 shuffle 直接当下一次 P·V 的 A 操作数（G3 的关键技巧）。
:::

## 逐步演示

::: steps lane 5 的一生：从 SMEM 到累加器
::: flow
cfg cw=200 rh=96 nw=170
n smem 0 0 "SMEM 16×16" "行跨度 16" k=dark
n addr 1 0 "lane 5 报地址" "矩阵 0 第 5 行 = 80" k=teal
n regs 2 0 "lane 5 的寄存器" "8 个 bf16" k=acc
n mma 3 0 "mma.sync" "和 B、C 一起" k=warn
n c 3 1 "lane 5 的 C" "(1,2)(1,3)(9,2)(9,3)" k=pass
e smem -> addr
e addr -> regs
e regs -> mma
e mma -> c
:::
--- step 坐标 {hl=regs}
lane 5：g = 1，t = 1。`a_coords(5)` = (1,2) (1,3) (9,2) (9,3) (1,10) (1,11) (9,10) (9,11)：行 1 和行 9，列 2–3 和 10–11。
--- step 报地址 {hl=smem,addr}
`a_tile_row_addrs(0, 16)`：lane 0..7 报矩阵 0（左上）的第 0..7 行 = 0, 16, …, 112；lane 8 报矩阵 1（左下）第 0 行 = 128；lane 16 报矩阵 2（右上）第 0 行 = 8；lane 24 报矩阵 3 第 0 行 = 136。lane 5 报的是 80（矩阵 0 第 5 行），**但它拿到的不是第 5 行**。
--- step 拿数据 {hl=addr,regs}
以 smem[i] = i 为例，ldmatrix.x4 后 lane 5 拿到 `[18 19 146 147 26 27 154 155]`：18 = 第 1 行第 2 列，146 = 第 9 行第 2 列，26 = 第 1 行第 10 列……正好是它的 `a_coords`。lane 0 拿到 `[0 1 128 129 8 9 136 137]`。
--- step MMA {hl=regs,mma,c}
32 个 lane 的 a、b、c 拼成 16×16、16×8、16×8 三个矩阵，算 D = A·B + C，再按 `c_coords` 分回：lane 5 得到 D 的 (1,2) (1,3) (9,2) (9,3)。
:::

## 易错点

- **把 a1 和 a2 搞反。** PTX 规定 a1 是“下半”（行 g+8）而 a2 是“右半”（列 +8）。顺序错了 `test_lane_zero_and_lane_five_coordinates` 直接失败。
- **以为 lane l 拿到的就是自己报地址的那一行。** 分发规则是 `l/4` 行，和报地址的 `l%8` 无关；`test_ldmatrix_x1_distributes_rows_to_quads` 抓这个。
- **B 用了 A 的规律。** B 寄存器里的两个元素是同一列 n 的相邻两个 k（“col” 布局），所以 B 要按 N×K（B 的转置、k 连续）存放才能用非转置 ldmatrix；否则要 `ldmatrix.trans`（G3 读 V 时就是这样）。
- **x2 时用了 lane 16..31 的地址。** x2 只看 lane 0..15，其余地址被忽略（但必须是合法地址）。
- **在 mma 里按 lane 顺序直接 reshape。** fragment 不是按 lane 连续排列的，必须按坐标拼。

## 三级提示

::: hint 1 I/O 示例与关键问题
```python
a_coords(0)   # [(0,0),(0,1),(8,0),(8,1),(0,8),(0,9),(8,8),(8,9)]
a_coords(31)  # [(7,6),(7,7),(15,6),(15,7),(7,14),(7,15),(15,14),(15,15)]
b_coords(31)  # [(6,7),(7,7),(14,7),(15,7)]
c_coords(31)  # [(7,6),(7,7),(15,6),(15,7)]
a_tile_row_addrs(0, 16)[:9]   # [0,16,32,48,64,80,96,112,128]
```

#### 关键问题
1. 每个 lane 拿 A 的 8 个元素、B 的 4 个、C 的 4 个。总数对得上吗（16·16、16·8、16·8）？
2. 一行 A（16 个元素）分布在哪几个 lane 上？
3. 为什么 C 布局和 A 布局“按行对齐”很重要？
4. B 如果是 K×N、n 连续地存在 SMEM 里，该用哪种 ldmatrix？
:::

::: hint 2 思路与常见坑
- 坐标函数：先算 `g, t`，A 按 `(0,0),(8,0),(0,8),(8,8)` 四个偏移各放两个相邻列。
- `mma_m16n8k16`：建三个零矩阵，按坐标把 32 行数据散进去，`D = A @ B + C`，再按 `c_coords` 收回来。
- `ldmatrix`：外层循环矩阵 j，取 8 行 `smem[addr : addr+8]`；内层循环 lane：`r = l // 4`，`c = 2*(l % 4)`。
- 地址函数：A 的矩阵 j：行偏移 `8*(j%2)`，列偏移 `8*(j//2)`；B 的矩阵 j：列偏移 `8*j`（只有 2 个矩阵）。
:::

::: hint 3 详解：答案
::: answer 关键问题答案
1. 32 × 8 = 256 = 16×16；32 × 4 = 128 = 16×8，两者都对得上，没有重复也没有遗漏（`test_each_element_owned_by_exactly_one_lane`）。
2. 第 g 行在 lane 4g..4g+3 上，每个 lane 4 个元素（列 2t、2t+1、2t+8、2t+9）。所以 FA 里“求一行的最大值”只要在 quad 内做两次 `__shfl_xor_sync`（1 和 2）。
3. S = QKᵀ 的输出在 C 布局，P = exp(S − m) 要作为下一次 P·V 的 A 操作数。两者行对齐，只需把 c0..c3 两两打包成 bf16（`pack_bf16`），零 shuffle。FA2 源码里的 [[fa:csrc/flash_attn/src/utils.h@convert_layout_acc_Aregs|convert_layout_acc_Aregs]] 做的就是这件事。
4. `ldmatrix.trans`：它把 8×8 子块转置后再分发，lane 拿到的是同一列相邻两行，也就是 B 布局。G3 读 V（`[key][d]`）时用的就是 `ldmatrix_x4_trans`，对应 CuTe 的 [[cutlass:include/cute/arch/copy_sm75.hpp@SM75_U16x8_LDSM_T|SM75_U16x8_LDSM_T]]。
:::
:::

## 参考解

::: solution solutions/a2_mma_fragments.py
{{code: solutions/a2_mma_fragments.py}}
:::

## 面试官追问 & 对照源码

- **“Hopper 的 wgmma 还需要 ldmatrix 吗？”** 不需要：A、B 用 64-bit SMEM 描述符直接从 SMEM 读（`desc_a / desc_b`），只有累加器在寄存器里。描述符里要写 swizzle 模式，所以 A3 / B3 的 swizzle 必须和 TMA 写入一致。
- **“Blackwell 呢？”** `tcgen05.mma` 由一个线程发起，A/B 在 SMEM（A 也可以在 TMEM），D 在 TMEM：fragment 布局这件事从寄存器搬到了 TMEM 的 lane/列上（C1）。
- **“m16n8k16 的 FP8 版本？”** sm89 / sm120 有 `mma.sync … .e4m3`，k = 32：同样的 16×8 输出，每个寄存器装 4 个 FP8。sm120 的 block-scaled 版本见 [[cutlass:include/cute/arch/mma_sm120.hpp@SM120_16x8x32_TN|SM120_16x8x32_TN]]。
- **“为什么叫 TN？”** BLAS 的列主序视角下 A 转置（T）、B 不转置（N）；换成行主序就是 “A 是 M×K 行主序、B 是 N×K 行主序”，两边都是 K 连续——ldmatrix 非转置就能直接读，所以 TN 是最快的布局。
