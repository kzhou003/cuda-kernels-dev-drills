"""D1 · Online softmax, (O, LSE) states and merging split-KV partials — reference solution.

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
    m, l = -np.inf, 0.0
    x = np.asarray(x, dtype=np.float64)
    for i in range(0, len(x), block):
        blk = x[i:i + block]
        m_new = max(m, blk.max(initial=-np.inf))
        if m_new == -np.inf:
            continue            # 这块全是 −inf：什么都不变（否则 exp(−inf − (−inf)) = NaN）
        # 旧的和要换到新的基准：Σ exp(x − m_old) · exp(m_old − m_new)。
        l = l * np.exp(m - m_new) + np.exp(blk - m_new).sum()
        m = m_new
    return m, l


def online_attention(Q, K, V, scale: float, block: int, mask=None):
    """FA 的核心递推：按 key 块更新 m、l、acc；最后 O = acc / l。"""
    Q, K, V = (np.asarray(a, dtype=np.float64) for a in (Q, K, V))
    nq, nk = Q.shape[0], K.shape[0]
    m = np.full(nq, -np.inf)
    l = np.zeros(nq)
    acc = np.zeros((nq, V.shape[1]))
    for j in range(0, nk, block):
        S = scale * Q @ K[j:j + block].T
        if mask is not None:
            S = np.where(mask[:, j:j + block], S, -np.inf)
        m_new = np.maximum(m, S.max(axis=1, initial=-np.inf))
        # m_new 仍是 −inf 的行（到目前为止全被遮住）用 0 做基准，避免 −inf − (−inf)。
        # FA2 源码里叫 “max_scaled = max == -INF ? 0 : max”（softmax.h），同一个技巧。
        safe = np.where(np.isfinite(m_new), m_new, 0.0)
        alpha = np.exp(np.where(np.isfinite(m), m - safe, -np.inf))
        P = np.exp(S - safe[:, None])
        l = l * alpha + P.sum(axis=1)
        acc = acc * alpha[:, None] + P @ V[j:j + block]
        m = m_new
    ok = l > 0
    O = np.where(ok[:, None], acc / np.where(ok, l, 1.0)[:, None], 0.0)
    lse = np.where(ok, m + np.log(np.where(ok, l, 1.0)), -np.inf)
    # 为什么返回 lse 而不是 (m, l)：lse 一个数就够合并（见 merge_states），反向传播也只要它
    # （FA2 的 softmax_lse）。O 已经归一化，所以每段的结果本身就是“这段 key 上的 attention”。
    return O, lse


def merge_states(o_a, lse_a, o_b, lse_b):
    """两段 (O, LSE) 合并成一段。"""
    o_a, o_b = np.asarray(o_a, dtype=np.float64), np.asarray(o_b, dtype=np.float64)
    lse_a, lse_b = np.asarray(lse_a, dtype=np.float64), np.asarray(lse_b, dtype=np.float64)
    lse = np.logaddexp(lse_a, lse_b)            # −inf 与 −inf 合并仍是 −inf，不会出 NaN
    fin = np.isfinite(lse)
    base = np.where(fin, lse, 0.0)
    wa = np.where(np.isfinite(lse_a), np.exp(lse_a - base), 0.0)
    wb = np.where(np.isfinite(lse_b), np.exp(lse_b - base), 0.0)
    # 权重 wa + wb = 1：每段的 O 是那段 key 上的加权平均，合起来就是按“质量”exp(lse) 再平均一次。
    o = wa[:, None] * o_a + wb[:, None] * o_b
    return np.where(fin[:, None], o, 0.0), lse


def split_kv_attention(Q, K, V, scale: float, num_splits: int, mask=None):
    """把 key 切成 num_splits 段分别算，再依次合并。"""
    idx = np.array_split(np.arange(K.shape[0]), num_splits)
    o = lse = None
    for part in idx:
        sub_mask = None if mask is None else mask[:, part]
        o_p, lse_p = online_attention(Q, K[part], V[part], scale, max(len(part), 1), sub_mask)
        if o is None:
            o, lse = o_p, lse_p
        else:
            o, lse = merge_states(o, lse, o_p, lse_p)
    # decode 时 nq 很小（1 个 token × head 数），只按 query 并行喂不饱 132 个 SM；
    # 切开 key 让多个 CTA 各算一段，再用一个小 kernel 合并——这就是 FlashDecoding。
    return o, lse
