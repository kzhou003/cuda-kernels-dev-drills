"""D1 · Online softmax, (O, LSE) states and merging split-KV partials — your implementation (run ./check.sh d1).

FlashAttention 的全部魔法都建立在一个递推上：softmax 不必等看完整行再算。维护“到目前为止的
最大值 m、指数和 l、未归一化的输出 acc”，每来一块新的 key 就把旧的 l、acc 乘 exp(m_old − m_new)
修正一下。同一个代数还给出“两段 attention 结果怎么合并”：每段带上 log-sum-exp（LSE），
合并就是按 exp(lse_i − lse) 加权。split-KV decode（FlashDecoding、FlashInfer 的 merge_state）
和 FA3/FA4 的 combine kernel 都是这么做的。

约定：自然对数；softmax_scale 由调用者给出；mask 为 bool 数组（True = 保留）。被完全遮住的行
（没有任何保留的 key）输出全 0、lse = −inf，而且不能产生 NaN。

- online_softmax_stats(x, block)：1D x 按 block 个一组顺序处理，返回 (m, l)，m = max(x)，
  l = Σ exp(x − m)。x 为空或全为 −inf 时返回 (−inf, 0.0)。
- online_attention(Q, K, V, scale, block, mask=None)：Q (nq, d)、K (nk, d)、V (nk, dv)。对 key 按 block 分块，
  用 m / l / acc 递推，返回 (O (nq, dv), lse (nq,))，O = softmax(scale·QKᵀ 遮罩后) V，lse = log Σ exp(...)。
- merge_states(o_a, lse_a, o_b, lse_b)：逐行合并两段结果，返回 (o, lse)；两边都是 −inf 的行得 (0, −inf)。
- split_kv_attention(Q, K, V, scale, num_splits, mask=None)：key 用 np.array_split 切成 num_splits 段
  （段可以为空），每段 online_attention（block 取该段长度，至少 1），再从左到右依次 merge_states。

API
    online_softmax_stats(x, block) -> tuple[float, float]
    online_attention(Q, K, V, scale, block, mask=None) -> tuple[np.ndarray, np.ndarray]
    merge_states(o_a, lse_a, o_b, lse_b) -> tuple[np.ndarray, np.ndarray]
    split_kv_attention(Q, K, V, scale, num_splits, mask=None) -> tuple[np.ndarray, np.ndarray]
"""
from __future__ import annotations

import numpy as np


def online_softmax_stats(x, block: int) -> tuple[float, float]:
    """分块扫描一遍，得到最大值和指数和。"""
    raise NotImplementedError("online_softmax_stats")


def online_attention(Q, K, V, scale: float, block: int, mask=None):
    """FA 的核心递推：按 key 块更新 m、l、acc；最后 O = acc / l。"""
    raise NotImplementedError("online_attention")


def merge_states(o_a, lse_a, o_b, lse_b):
    """两段 (O, LSE) 合并成一段。"""
    raise NotImplementedError("merge_states")


def split_kv_attention(Q, K, V, scale: float, num_splits: int, mask=None):
    """把 key 切成 num_splits 段分别算，再依次合并。"""
    raise NotImplementedError("split_kv_attention")
