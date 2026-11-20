lede: Tensor Core 再快，操作数也要先从 SMEM 读出来。SMEM 有 32 个 bank，摆放方式不对，一次读取会被拆成 8 次。swizzle 用一个 XOR 把同一“列块”的 8 行打散到 8 组 bank 上，CuTe、TMA、wgmma 描述符全都靠它。
# A3 · SMEM bank conflict 与 CuTe Swizzle<B,M,S>

## 题面速览

- **在系统中的位置**：CuTe 的 [[cutlass:include/cute/swizzle.hpp@Swizzle|Swizzle<BBits, MBase, SShift>]] 用在每一个 SMEM layout 上（`composition(Swizzle<3,3,3>{}, …)`）；TMA 描述符里的 `CU_TENSOR_MAP_SWIZZLE_128B`（B3）和 wgmma / tcgen05 的 SMEM 描述符里的 swizzle 模式是同一个函数。G1 / G2 / G3 在 5090 上用的是另一种办法：padding。
- **你要实现**：`wavefronts_32bit`（普通 32-bit 访问的冲突数）、`ldmatrix_wavefronts`（ldmatrix 每 8 行一个相位）、`swizzle`、`swizzled_offset`、`padded_offset`。
- **难点在这几条规则**：
  1. 冲突只算**同一 bank 上不同的字**；32 个线程读同一个字是广播，1 次。
  2. ldmatrix 的一个 8×8 矩阵 = 8 行 × 16 B = 128 B，正好绕 32 个 bank 一圈；8 个 16 B 块落在 8 个不同的“槽”（每槽 4 个 bank）才是 1 次。
  3. `Swizzle<B,M,S>`：保留低 M 位，把第 `[M+S, M+S+B)` 位右移 S 位 XOR 到第 `[M, M+B)` 位。
  4. XOR 是自己的逆，所以写入和读出用同一个函数。

**验收要点**：连续字无冲突 · 广播不算冲突 · 32×32 float 读一列 32 路冲突 · padding 1 个 float 消除 · 步长 2 是 2 路 · 每行 128 B 的 tile 用 ldmatrix 是 8 路 · 对齐检查 · swizzle 已知值 · swizzle 是置换且自逆 · 16 B 向量不被拆开 · swizzle 后每个列块都无冲突 · 行读也无冲突。

## 直觉与类比

::: callout analogy 停车场的错位车位
SMEM 像一个有 8 排车位的停车场（8 个槽，每槽 4 个 bank），每辆车（16 B 块）按“第几列”停在固定的排上。按行存的 tile 里，每一行的第 0 个块都停在第 0 排——ldmatrix 一次要取 8 行的第 0 块，8 辆车全挤在一排出口，只能排队一辆一辆开出去（8 个 wavefront）。swizzle 的做法是**第 r 行的车错开 r 个位置停**（块号 XOR 行号）：取同一列块时，8 辆车分别在 8 个不同的出口，一次全部开出去。padding 是另一种办法：每行多空几个车位，让下一行自然错开——简单，但浪费车位，而且 TMA 不认识这种停法。
:::

## 图解：一个 bf16 tile，每行 64 个元素（128 B）

::: flow Swizzle<3,3,3> 怎么读一个元素偏移
cfg cw=200 rh=96 nw=170
n off 0 0 "元素偏移 o" "= 行·64 + 列" k=dark
n low 1 0 "低 3 位：保留" "16 B 内的 8 个元素" k=pass
n mid 1 1 "第 3–5 位：块号" "c = 列 / 8" k=acc
n hi 2 1 "第 6–8 位：行号低 3 位" "r % 8" k=vio
n out 3 0.5 "新块号 = c ⊕ (r%8)" "8 行落在 8 个槽" k=teal
e off -> low
e off:b -> mid:l
e mid -> hi
e hi -> out
e low -> out
cap: 单位是元素：M = 3（8 个 bf16 = 16 B 不拆开，ldmatrix / cp.async 的 16 B 向量完整），B = 3（8 个块要打散），S = 3（钥匙在块号上面 3 位，即一行 64 个元素之上的行号）。TMA 的 128B swizzle 是同一个函数，只是按字节写成 Swizzle<3,4,3>。
:::

## 逐步演示

::: steps 8 行、每行 128 B 的 bf16 tile，ldmatrix 读第 0 个列块
::: flow
cfg cw=200 rh=90 nw=170
n row 0 0 "按行存放" "块 0 都在槽 0" k=todo
n pad 1 0 "padding +8" "每行 144 B" k=warn
n swz 2 0 "Swizzle<3,3,3>" "块 0 → 槽 r" k=pass
n res 3 0 "wavefront 数" "8 / 1 / 1" k=teal
e row -> pad
e pad -> swz
e swz -> res
:::
--- step 按行存放 {hl=row}
第 r 行第 0 块的字节地址 = 128·r，槽 = (128r / 16) % 8 = 0。8 行全在槽 0：`ldmatrix_wavefronts` = **8**。
{{cells: bad:0 bad:0 bad:0 bad:0 bad:0 bad:0 bad:0 bad:0}}
--- step padding 8 个元素 {hl=pad}
行跨度变成 72 个元素 = 144 B，槽 = 9r % 8 = r：8 个不同的槽，**1** 次。代价是 12.5% 的 SMEM，而且 TMA 不能按这个布局写。
{{cells: ok:0 ok:1 ok:2 ok:3 ok:4 ok:5 ok:6 ok:7}}
--- step swizzle {hl=swz}
第 r 行的块 0 被放到块 0 ⊕ r = r：同样 8 个不同的槽，**1** 次，一个字节都不浪费。第 3 行的 8 个块依次去了 `3 2 1 0 7 6 5 4`。
{{cells: ok:0 ok:1 ok:2 ok:3 ok:4 ok:5 ok:6 ok:7}}
--- step 普通 32-bit 访问 {hl=res}
另一个经典例子：32×32 的 float tile 读一列（转置）：`wavefronts_32bit` = **32**；每行多 1 个 float（ld = 33）后 = **1**。
:::

## 易错点

- **数线程数而不是数不同的字。** 广播会被误算成 32 路冲突（`test_broadcast_is_not_a_conflict`）。
- **ldmatrix 按 32-bit 规则算。** ldmatrix 每次处理 8 行（一个 128 B 相位），按 16 B 块和 8 个槽算；x4 是 4 个相位，最少 4 次（`test_ldmatrix_contiguous_matrix_is_one_wavefront`）。
- **swizzle 把 16 B 向量拆开了。** M 选小了（例如 bf16 用 M = 2），一个 cp.async 的 16 B 写入会落到两个不连续的位置。`test_swizzle_keeps_16_byte_vectors_contiguous` 抓这个。
- **掩码位置错一位。** 钥匙是第 `[M+S, M+S+B)` 位，不是 `[S, S+B)`。`test_swizzle_known_values` 里第 9 行要表现得像第 1 行。
- **写用 swizzle、读没用。** 两边必须是同一个函数；B3 里 TMA 写、wgmma 描述符读，靠的就是“约定同一个 swizzle 模式”。

## 三级提示

::: hint 1 I/O 示例与关键问题
```python
wavefronts_32bit([(r * 32 + 5) * 4 for r in range(32)])          # 32
wavefronts_32bit([padded_offset(r, 5, 32, 1) * 4 for r in range(32)])   # 1
ldmatrix_wavefronts([r * 128 for r in range(8)])                # 8
ldmatrix_wavefronts([2 * swizzled_offset(r, 0, 64, 3, 3, 3) for r in range(8)])  # 1
swizzle(64, 3, 3, 3)                                            # 72
```

#### 关键问题
1. 地址 a 的 bank 是多少？什么时候两个访问冲突？
2. ldmatrix 的 8 行为什么是“一组”？
3. `Swizzle<3,3,3>` 用在 bf16 上，M、B、S 各自对应 tile 的什么？
4. 换成 FP8（每个元素 1 字节），同样 128 B 一行，swizzle 参数怎么变？
:::

::: hint 2 思路与常见坑
- `wavefronts_32bit`：`defaultdict(set)`，键 = `(a//4) % 32`，值 = `{a//4}`；答案是最大集合的大小。
- `ldmatrix_wavefronts`：先校验（长度是 8 的倍数、地址 16 对齐），每 8 个一组，键 = `(a//16) % 8`，值 = `{a//16}`，每组取最大，再求和。
- `swizzle`：`mask = ((1 << B) - 1) << (M + S)`；`return o ^ ((o & mask) >> S)`。
- `swizzled_offset` / `padded_offset` 各一行。
:::

::: hint 3 详解：答案
::: answer 关键问题答案
1. bank = `(a // 4) % 32`。同一 wavefront 里两个访问落在同一 bank 且是**不同**的 4 字节字，就冲突。
2. ldmatrix 的每个 8×8 矩阵是 8 行 × 16 B = 128 B，硬件一次 SMEM 事务最多 128 B（32 bank × 4 B）。所以每个矩阵是一个相位，相位内 8 个 16 B 块要落在 8 个不同的 4-bank 组上。
3. M = 3：低 3 位 = 8 个 bf16 = 16 B（不拆开的向量）；B = 3：一行 128 B 有 8 个 16 B 块；S = 3：从块号再往上 3 位，就是“行号”（每行 64 个元素 = 2^6）。
4. FP8 一行 128 B 是 128 个元素，16 B = 16 个元素：Swizzle<3,4,3>（M = 4）。用字节偏移写就永远是 Swizzle<3,4,3>，这正是 TMA 128B swizzle 的定义。
:::
:::

## 参考解

::: solution solutions/a3_bank_swizzle.py
{{code: solutions/a3_bank_swizzle.py}}
:::

## 面试官追问 & 对照源码

- **“CUTLASS 怎么选 swizzle？”** 按 SMEM tile 的 K 方向字节数选最大的那个：K 方向 ≥ 128 B 用 128B swizzle（`Swizzle<3,4,3>`），64 B 用 `Swizzle<2,4,3>`，32 B 用 `Swizzle<1,4,3>`。见 [[cutlass:include/cute/swizzle.hpp@Swizzle|cute/swizzle.hpp]] 和 sm90 的 [[cutlass:include/cutlass/gemm/collective/builders/sm90_common.inl@ss_smem_selector|ss_smem_selector]]。
- **“epilogue 写 SMEM 也会冲突吗？”** 会：累加器的 C 布局（每线程 2 个相邻 float）按行写，每个 quad 写同一行，8 个 quad 写 8 行的同一列段——同样需要 swizzle 或 padding。
- **“5090 上为什么 G1 用 padding 不用 swizzle？”** 代码更好懂；5090 的 SMEM 预算（100 KB/SM）里 20% 的 padding 在 2 级流水下还放得下。写 TMA 版本时必须换成 swizzle。
- **“bank 冲突在 Nsight Compute 里看什么？”** `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld` 和 “Shared Load Wavefronts / Requests” 的比值。
