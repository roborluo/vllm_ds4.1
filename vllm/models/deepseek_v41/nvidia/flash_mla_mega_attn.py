# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 attention on FlashMLA's mega-attention kernel (SM100).

One kernel does Q RoPE, sparse attention, the inverse RoPE of the output and
its FP8 cast, and writes straight into the buffer ``wo_a`` consumes. So the
layer declares ``accepts_unnormed_unroped_query`` -- the kernel RoPEs Q itself,
leaving the fused KV insert only the zero-pad to the kernel's head count -- and
its ``_alloc_attn_out`` / ``_o_proj`` pair speaks QuantizedActivation instead
of bf16, since the output needs no rotation or quantization here.

``wq_b`` rows and ``wo_a`` columns are permuted once at load so the surrounding
GEMMs speak the kernel's chunk-interleaved layouts directly. A step's prefill
and decode segments write disjoint token ranges of one output buffer pair, so
a single ``wo_a`` einsum covers the whole step.
"""

from dataclasses import replace
from typing import TYPE_CHECKING, NamedTuple, cast

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.models.deepseek_v41.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
)
from vllm.models.deepseek_v41.common.ops.fused_layout import (
    WV_GROUP_SIZE,
    permute_wo_a_,
    permute_wq_b_,
)
from vllm.models.deepseek_v41.nvidia.flashmla import DeepseekV4FlashMLAAttention
from vllm.models.deepseek_v41.sparse_mla import (
    DeepseekV4FlashMLAMetadata,
    FlashMLAMegaAttnBackend,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum, get_tma_aligned_size
from vllm.utils.math_utils import round_up
from vllm.v1.attention.ops.flashmla import is_flashmla_sparse_supported
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

_HEAD_DIM_V = 512
_ROPE_DIM = 64
# The kernel always emits per-32 scales, whatever num_per_channels says.
_QUANT_GROUP = 32
# Fixed kernel conventions: GPT-J (non-neox) RoPE, TMA-aligned col-major
# scales, round-to-nearest, packed ue8m0 -- i.e. what DeepGEMM's fp8_einsum
# consumes. Q norm is off: the V4.1 layer norms qr before wq_b.
_ROPE_ARGS = (False, _ROPE_DIM)
_SF_ARGS = (_QUANT_GROUP, True, True, True)

# Decode always goes through the mega kernel: there is no batch size at which
# falling back to the split-KV decode pays, so there is no minimum-token
# threshold and no fallback path to maintain. What decides whether this layer
# is worth selecting at all is TP, not the batch size.
#
# Measured on one GB300 with benchmarks/kernels/benchmark_dsv41_mega_attn.py,
# CUDA graphs on, counting every per-step op each path runs (64-head model,
# topk_swa=128, topk_extra=512, NVFP4 compressed cache), as mega / split-KV
# microseconds:
#
#     s_q                     1      32     128     256     512
#     TP1 (64 live heads)  29/29   31/31   31/40   53/72   96/140
#     TP2 (32 live)        29/29   31/30   35/35   61/63  110/119
#     TP4 (16 live)        29/29   31/29   34/34   59/58  109/109
#     TP8 (8 live)         29/29   31/29   33/33   59/57  107/104
#
# The kernel absorbs the inverse RoPE and the FP8 cast, whose cost scales with
# *live* heads, while its own cost scales with *padded* heads -- always 64,
# since FlashMLA's FP8 decode takes only h_q in {64, 128}. So it wins at TP1
# (up to 1.45x, and TP1 is also where the fused insert skips the Q pad
# entirely) and from TP2 down the two cancel to within a few percent, with a
# fixed ~2 us penalty at small s_q where the padded-head cost dominates.
# Measure before assuming a win at high TP -- and measure under CUDA graphs:
# in eager mode launch overhead swamps both paths and overstates the fused one
# several-fold.


# ---------------------------------------------------------------------------
# bofengl (attn agent, 10/12): opt-in swap-AB small-batch decode attention.
#
# VLLM_BOFENGL_SWAPAB_ATTN=1 replaces the mega kernel on decode / verify / draft
# segments with <= VLLM_BOFENGL_SWAPAB_MAX_TOKENS query tokens and 16 or 32 live
# heads by a port of SGLang's swap-AB Gluon kernel (Apache-2.0,
# sglang/kernels/ops/attention/dsv4/decode_attention_sm100{,_gluon}.py): the
# live heads sit on the tcgen05 MMA N dimension instead of being padded to 64
# on M, one CTA per (query token, 64-slot KV tile), then a Triton combine.
# Adapted to vLLM's records and contract:
#   * KV: V4.1 fp8 SWA record (528 B: 512 e4m3 + 16 ue8m0 per 32) and V4.1
#     NVFP4 compressed record (288 B: 256 B e2m1 + 32 e4m3 per 16) instead of
#     SGLang's V4 record (584 B);
#   * Q: the raw wq_b output in the mega kernel's chunk-interleaved, head-padded
#     layout; GPT-J RoPE on the last 64 dims is applied in the kernel;
#   * output: the combine folds the attention sink (once), merges the split-KV
#     partials, applies the inverse RoPE, casts to MXFP8 (ue8m0 per 32) and
#     stores data + packed scales in the o-fused layout wo_a was permuted for.
# VLLM_BOFENGL_SWAPAB_ATTN=2 is a shadow/debug mode: the mega kernel still
# produces the output; eager (non-capturing) calls also run the swap-AB path
# and log the dequantized difference.
# ---------------------------------------------------------------------------
import os as _os

_SWAPAB_MODE = _os.environ.get("VLLM_BOFENGL_SWAPAB_ATTN", "0")
_SWAPAB_MAX_TOKENS = int(_os.environ.get("VLLM_BOFENGL_SWAPAB_MAX_TOKENS", "16"))
_SWAPAB_BT = 64
# Tunables. Defaults reproduce the mega
# kernel's output bit-exactly on real inputs (shadow mode, median call):
# COMPENSATE: second P.V MMA on the bf16 residual of P; PART_BF16=0: fp32
# split-KV partials. (1/0 -> 0/1 is ~0.5-0.9 us faster per call but differs
# from mega by ~1% fro, an FP8-output-level perturbation.)
# VEC: 3-D gather with one scale load per scale group and native e2m1 upcast.
# SWA_ONLY: also take SWA-only layers (2 CTAs/token; mega is faster there).
_SWAPAB_COMPENSATE = _os.environ.get("VLLM_BOFENGL_SWAPAB_COMPENSATE", "1") == "1"
_SWAPAB_PART_BF16 = _os.environ.get("VLLM_BOFENGL_SWAPAB_PART_BF16", "0") == "1"
_SWAPAB_VEC = _os.environ.get("VLLM_BOFENGL_SWAPAB_VEC", "1") == "1"
_SWAPAB_SWA_ONLY = _os.environ.get("VLLM_BOFENGL_SWAPAB_SWA_ONLY", "0") == "1"
# DIRECT: store partials straight from the TMEM register layout (no smem
# transpose). PDL: launch the combine with programmatic dependent launch.
_SWAPAB_DIRECT = _os.environ.get("VLLM_BOFENGL_SWAPAB_DIRECT", "1") == "1"
_SWAPAB_PDL = _os.environ.get("VLLM_BOFENGL_SWAPAB_PDL", "1") == "1"
_SWAPAB_SHADOW_LOG_LIMIT = int(_os.environ.get("VLLM_BOFENGL_SWAPAB_SHADOW_LOG", "400"))
_swapab_shadow_logged = 0

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.nvidia.blackwell import (
        TensorMemoryLayout,
        allocate_tensor_memory,
        fence_async_shared,
        mbarrier,
        tcgen05_commit,
        tcgen05_mma,
    )

    try:  # Triton 3.7.x
        from triton.experimental.gluon.language.nvidia.blackwell import (
            get_tmem_reg_layout,
        )
    except ImportError:  # Triton 3.8 moved it onto the descriptor type
        from triton.experimental.gluon.language.nvidia.blackwell import (
            tensor_memory_descriptor_type,
        )
        from triton.runtime.jit import constexpr_function

        @constexpr_function
        def get_tmem_reg_layout(element_ty, shape, layout, num_warps):
            ty = tensor_memory_descriptor_type(element_ty, shape, layout, shape)
            return ty.get_reg_layout(num_warps=num_warps)

    _HAS_GLUON = True
except Exception:  # noqa: BLE001 - opt-in path only; reported when requested
    _HAS_GLUON = False

from vllm.triton_utils import tl, triton  # noqa: E402

if _HAS_GLUON:

    @gluon.jit
    def _at_load_kv(
        CACHE,
        ids,
        valid,
        PAGE: gl.constexpr,
        STRIDE: gl.constexpr,
        KV_LAYOUT: gl.constexpr,
        FP4: gl.constexpr,
    ):
        # Page = PAGE data rows then PAGE scale rows (V4.1 records).
        d = gl.arange(0, 512, gl.SliceLayout(0, KV_LAYOUT))
        base = (ids // PAGE).to(gl.int64)[:, None] * STRIDE
        slot = (ids % PAGE)[:, None]
        mask = valid[:, None] & (d[None, :] >= 0)
        if FP4:
            # 512 e2m1 packed two per byte (even element low nibble), then 32
            # e4m3 scales of 16 values.
            byte = gl.load(CACHE + base + slot * 256 + d[None, :] // 2, mask, 0).to(
                gl.int32
            )
            code = (byte >> ((d[None, :] & 1) * 4)) & 15
            # e2m1 -> fp16 bits (s, e1e0, m at the fp16 exponent/mantissa
            # top) is the value * 2^-14, denormal for 0.5; the 2^14 goes in
            # the scale.
            h16 = (((code & 8) << 12) | ((code & 7) << 9)).to(gl.int16)
            val = h16.to(gl.float16, bitcast=True).to(gl.float32)
            sf = gl.load(
                CACHE + base + PAGE * 256 + slot * 32 + d[None, :] // 16, mask, 0
            )
            scale = sf.to(gl.float8e4nv, bitcast=True).to(gl.float32) * 16384.0
        else:
            # 512 e4m3 (RoPE included), then 16 ue8m0 scales of 32 values.
            bits8 = gl.load(CACHE + base + slot * 512 + d[None, :], mask, 0)
            val = bits8.to(gl.float8e4nv, bitcast=True).to(gl.float32)
            exponent = gl.load(
                CACHE + base + PAGE * 512 + slot * 16 + d[None, :] // 32, mask, 0
            ).to(gl.int32)
            scale = gl.where(exponent == 0, 0x00400000, exponent << 23).to(
                gl.float32, bitcast=True
            )
        return (val * scale).to(gl.bfloat16)

    @gluon.jit
    def _at_load_kv_vec(
        CACHE,
        ids,
        valid,
        PAGE: gl.constexpr,
        STRIDE: gl.constexpr,
        KV_LAYOUT: gl.constexpr,
        FP4: gl.constexpr,
        BT: gl.constexpr,
    ):
        # [BT, groups, bytes-per-group] gather: one scale load per thread and
        # scale group, then reshape to [BT, 512] (same thread mapping as
        # KV_LAYOUT, so the final convert is register-trivial).
        if FP4:
            L3: gl.constexpr = gl.BlockedLayout([1, 1, 4], [4, 4, 2], [4, 1, 1], [2, 1, 0])
            LV: gl.constexpr = gl.BlockedLayout([1, 1, 8], [4, 4, 2], [4, 1, 1], [2, 1, 0])
            NG: gl.constexpr = 32
            GB: gl.constexpr = 8
            NS: gl.constexpr = 32
        else:
            L3: gl.constexpr = gl.BlockedLayout([1, 1, 8], [4, 2, 4], [4, 1, 1], [2, 1, 0])
            LV: gl.constexpr = gl.BlockedLayout([1, 1, 8], [4, 2, 4], [4, 1, 1], [2, 1, 0])
            NG: gl.constexpr = 16
            GB: gl.constexpr = 32
            NS: gl.constexpr = 16
        LR: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, L3))
        LG: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, L3))
        LB: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, L3))
        ids3 = gl.convert_layout(ids, LR)
        valid3 = gl.convert_layout(valid, LR)
        base = (ids3 // PAGE).to(gl.int64) * STRIDE + (ids3 % PAGE) * (NG * GB)
        gi = gl.arange(0, NG, LG)
        bi = gl.arange(0, GB, LB)
        off = base[:, None, None] + (gi[None, :, None] * GB + bi[None, None, :])
        m3 = valid3[:, None, None] & (gi[None, :, None] >= 0) & (bi[None, None, :] >= 0)
        raw = gl.load(CACHE + off, m3, 0)
        LSR: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, LV))
        LSG: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, LV))
        ids_s = gl.convert_layout(ids, LSR)
        valid_s = gl.convert_layout(valid, LSR)
        sg = gl.arange(0, NG, LSG)
        soff = ((ids_s // PAGE).to(gl.int64) * STRIDE + PAGE * (NG * GB) + (ids_s % PAGE) * NS)[:, None] + sg[None, :]
        smask = valid_s[:, None] & (sg[None, :] >= 0)
        sraw = gl.load(CACHE + soff, smask, 0)
        if FP4:
            scale = sraw.to(gl.float8e4nv, bitcast=True).to(gl.float32)
            vals = gl.convert_layout(gl.fp4_to_fp(raw, gl.bfloat16, 2), LV).to(gl.float32)
        else:
            e = sraw.to(gl.int32)
            scale = gl.where(e == 0, 0x00400000, e << 23).to(gl.float32, bitcast=True)
            vals = raw.to(gl.float8e4nv, bitcast=True).to(gl.float32)
        kv3 = (vals * scale[:, :, None]).to(gl.bfloat16)
        return gl.convert_layout(gl.reshape(kv3, [BT, 512]), KV_LAYOUT)

    @gluon.jit
    def _at_partial(
        Q,
        K,
        E,
        IDX,
        EI,
        L,
        EL,
        POS,
        CS,
        PART,
        MAX,
        SUM,
        KTOKENS,
        ETOKENS,
        QS: gl.constexpr,
        HP: gl.constexpr,
        IS: gl.constexpr,
        EIS: gl.constexpr,
        CSS: gl.constexpr,
        KP: gl.constexpr,
        KS: gl.constexpr,
        EP: gl.constexpr,
        ES: gl.constexpr,
        NK: gl.constexpr,
        NE: gl.constexpr,
        NT: gl.constexpr,
        KT: gl.constexpr,
        BT: gl.constexpr,
        H: gl.constexpr,
        SCALE: gl.constexpr,
        E_FP4: gl.constexpr,
        COMPENSATE: gl.constexpr,
        DEBUG: gl.constexpr = 0,
        VEC: gl.constexpr = 0,
        DIRECT: gl.constexpr = 0,
    ):
        gl.static_assert((H == 16 or H == 32) and (BT == 64 or BT == 128))
        b, t = gl.program_id(0), gl.program_id(1)
        kv_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
        n = gl.arange(0, BT, gl.SliceLayout(1, kv_layout))
        if t < KT:
            at = t * BT + n
            length = gl.load(L + b)
            ids = gl.load(IDX + b * IS + at, at < NK, -1)
            valid = (at < NK) & (at < length) & (ids >= 0) & (ids < KTOKENS)
            if VEC:
                kv = _at_load_kv_vec(
                    K, gl.maximum(ids, 0), valid, KP, KS, kv_layout, False, BT
                )
            else:
                kv = _at_load_kv(
                    K, gl.maximum(ids, 0), valid, KP, KS, kv_layout, False
                )
        else:
            at = (t - KT) * BT + n
            length = gl.load(EL + b)
            ids = gl.load(EI + b * EIS + at, at < NE, -1)
            valid = (at < NE) & (at < length) & (ids >= 0) & (ids < ETOKENS)
            if VEC:
                kv = _at_load_kv_vec(
                    E, gl.maximum(ids, 0), valid, EP, ES, kv_layout, E_FP4, BT
                )
            else:
                kv = _at_load_kv(
                    E, gl.maximum(ids, 0), valid, EP, ES, kv_layout, E_FP4
                )
        if DEBUG == 1:  # timing probe only: no KV gather / dequant
            kv = gl.zeros([BT, 512], gl.bfloat16, kv_layout)
        # Q: chunk-interleaved head-padded layout, unroped -> GPT-J RoPE on the
        # last 64 dims (fp32), then bf16 for the MMA.
        qh = gl.arange(0, H, gl.SliceLayout(1, kv_layout))
        qd = gl.arange(0, 512, gl.SliceLayout(0, kv_layout))
        q_base = Q + b.to(gl.int64) * QS
        qoff = (qd[None, :] // 16) * (HP * 16) + qh[:, None] * 16 + (qd[None, :] % 16)
        q = gl.load(q_base + qoff).to(gl.float32)
        qd2 = qd ^ 1
        poff = (qd2[None, :] // 16) * (HP * 16) + qh[:, None] * 16 + (qd2[None, :] % 16)
        is_rope = qd >= 448
        rmask = is_rope[None, :] & (qh[:, None] >= 0)
        qp = gl.load(q_base + poff, rmask, 0.0).to(gl.float32)
        pos = gl.load(POS + b).to(gl.int64)
        fi = gl.maximum((qd - 448) // 2, 0)
        cosv = gl.load(CS + pos * CSS + fi, is_rope, 1.0).to(gl.float32)
        sinv = gl.load(CS + pos * CSS + 32 + fi, is_rope, 0.0).to(gl.float32)
        sgn = gl.where((qd & 1) == 0, -1.0, 1.0)
        q = q * cosv[None, :] + (sgn * sinv)[None, :] * qp
        q = q.to(gl.bfloat16)

        q_smem = gl.allocate_shared_memory(
            gl.bfloat16,
            [H, 512],
            gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16),
            value=q,
        )
        kv_smem = gl.allocate_shared_memory(
            gl.bfloat16,
            [BT, 512],
            gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16),
            value=kv,
        )
        score_tmem = allocate_tensor_memory(
            gl.float32, [BT, H], TensorMemoryLayout(block=(BT, H), col_stride=1)
        )
        bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(bar, count=1)
        fence_async_shared()
        tcgen05_mma(kv_smem, q_smem.permute((1, 0)), score_tmem, use_acc=False)
        tcgen05_commit(bar)
        mbarrier.wait(bar, phase=0)
        score_layout: gl.constexpr = get_tmem_reg_layout(
            gl.float32, (BT, H), TensorMemoryLayout(block=(BT, H), col_stride=1), 4
        )
        scores = score_tmem.load(score_layout) * SCALE
        valid = gl.convert_layout(valid, gl.SliceLayout(1, score_layout))
        scores = gl.where(valid[:, None], scores, -float("inf"))
        mx = gl.max(scores, 0)
        mx = gl.where(mx == -float("inf"), 0.0, mx)
        prob = gl.exp(scores - mx[None, :])
        denom = gl.sum(prob, 0)
        p_hi = prob.to(gl.bfloat16)
        P_SWIZZLE: gl.constexpr = 2 * H
        p_smem = gl.allocate_shared_memory(
            gl.bfloat16,
            [BT, H],
            gl.NVMMASharedLayout(swizzle_byte_width=P_SWIZZLE, element_bitwidth=16),
            value=p_hi,
        )
        out_tmem = allocate_tensor_memory(
            gl.float32, [512, H], TensorMemoryLayout(block=(128, H), col_stride=1)
        )
        fence_async_shared()
        tcgen05_mma(kv_smem.permute((1, 0)), p_smem, out_tmem, use_acc=False)
        tcgen05_commit(bar)
        mbarrier.wait(bar, phase=1)
        if COMPENSATE:
            p_lo = (prob - p_hi.to(gl.float32)).to(gl.bfloat16)
            p_smem.store(p_lo)
            fence_async_shared()
            tcgen05_mma(kv_smem.permute((1, 0)), p_smem, out_tmem, use_acc=True)
            tcgen05_commit(bar)
            mbarrier.wait(bar, phase=0)
        mbarrier.invalidate(bar)
        out_layout: gl.constexpr = get_tmem_reg_layout(
            gl.float32, (512, H), TensorMemoryLayout(block=(128, H), col_stride=1), 4
        )
        value = out_tmem.load(out_layout)
        if DIRECT:
            # out_layout: a thread owns rows d of [512, H]; for each h the warp
            # writes 32 consecutive d -> coalesced without a transpose.
            dd = gl.arange(0, 512, gl.SliceLayout(1, out_layout))
            hh = gl.arange(0, H, gl.SliceLayout(0, out_layout))
            gl.store(
                PART + ((b * NT + t) * H + hh[None, :]) * 512 + dd[:, None],
                value.to(PART.dtype.element_ty),
            )
        else:
            hd_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
            value_hd = gl.convert_layout(value.permute((1, 0)), hd_layout)
            h = gl.arange(0, H, gl.SliceLayout(1, hd_layout))
            d = gl.arange(0, 512, gl.SliceLayout(0, hd_layout))
            gl.store(
                PART + ((b * NT + t) * H + h[:, None]) * 512 + d[None, :],
                value_hd.to(PART.dtype.element_ty),
            )
        stat_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
        hs = gl.arange(0, H, stat_layout)
        gl.store(MAX + (b * NT + t) * H + hs, gl.convert_layout(mx, stat_layout))
        gl.store(SUM + (b * NT + t) * H + hs, gl.convert_layout(denom, stat_layout))


@triton.jit
def _at_combine(
    PART,
    MAX,
    SUM,
    SINK,
    OUT,
    OUT_SCALE,
    CS,
    POS,
    OUT_ST,
    OUT_SG,
    SC_ST,
    SC_SG,
    SC_SK,
    CSS: tl.constexpr,
    NT: tl.constexpr,
    ST: tl.constexpr,
    H: tl.constexpr,
    BD: tl.constexpr,
    GS: tl.constexpr,
    EPS: tl.constexpr,
    BF16_ROUND: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    b, h, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    t = tl.arange(0, ST)
    d = tile * BD + tl.arange(0, BD)
    sink = tl.load(SINK + h)
    pos = tl.load(POS + b).to(tl.int64)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()  # partials of the preceding swap-AB kernel
    den = tl.load(SUM + (b * NT + t) * H + h, t < NT, 0.0)
    mx = tl.load(MAX + (b * NT + t) * H + h, t < NT, 0.0)
    mx = tl.where(den > 0, mx, -float("inf"))
    m = tl.maximum(tl.max(mx, 0), sink)
    m = tl.where(tl.abs(m) == float("inf"), 0.0, m)
    factor = tl.exp(mx - m)
    denominator = tl.sum(den * factor, 0) + tl.exp(sink - m)
    vals = tl.load(
        PART + ((b * NT + t[:, None]) * H + h) * 512 + d[None, :],
        t[:, None] < NT,
        0.0,
    )
    out = tl.sum(vals.to(tl.float32) * factor[:, None], 0) / denominator
    out = tl.where((denominator > 0) & (sink != float("inf")), out, 0.0)
    if BF16_ROUND:
        out = out.to(tl.bfloat16).to(tl.float32)
    # Inverse GPT-J RoPE on the last 64 dims at the query position.
    is_rope = d >= 448
    fi = tl.maximum((d - 448) // 2, 0)
    cosv = tl.load(CS + pos * CSS + fi, is_rope, 1.0).to(tl.float32)
    sinv = tl.load(CS + pos * CSS + 32 + fi, is_rope, 0.0).to(tl.float32)
    partner = tl.reshape(tl.flip(tl.reshape(out, (BD // 2, 2)), 1), (BD,))
    sgn = tl.where((d & 1) == 0, 1.0, -1.0)
    x = out * cosv + sgn * sinv * partner
    # MXFP8: one ue8m0 scale per 32 values, round-to-nearest e4m3.
    NC: tl.constexpr = BD // 32
    x2 = tl.reshape(x, (NC, 32))
    amax = tl.maximum(tl.max(tl.abs(x2), 1), EPS)
    sc = tl.exp2(tl.ceil(tl.log2(amax * (1.0 / 448.0))))
    qv = tl.clamp(x2 / sc[:, None], -448.0, 448.0).to(tl.float8e4nv)
    # o-fused layout: fused[(c * GS + hg) * 32 + j] = std[hg * 512 + c * 32 + j].
    c = tile * NC + tl.arange(0, NC)
    g = h // GS
    hg = h % GS
    j = tl.arange(0, 32)
    off = b.to(tl.int64) * OUT_ST + g * OUT_SG + (c[:, None] * GS + hg) * 32 + j[None, :]
    tl.store(OUT + off, qv)
    e8 = ((sc.to(tl.int32, bitcast=True) >> 23) & 0xFF).to(tl.uint8)
    k = c * GS + hg
    sbyte = (b.to(tl.int64) * SC_ST + g * SC_SG + (k // 4) * SC_SK) * 4 + (k % 4)
    tl.store(OUT_SCALE.to(tl.pointer_type(tl.uint8)) + sbyte, e8)


def swapab_supported(n_live_heads: int) -> bool:
    return _HAS_GLUON and n_live_heads in (16, 32)


def swapab_mega_decode(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_idx: torch.Tensor,
    swa_len: torch.Tensor,
    extra_cache: torch.Tensor | None,
    extra_idx: torch.Tensor | None,
    extra_len: torch.Tensor | None,
    positions_int32: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    sink: torch.Tensor,
    n_live_heads: int,
    softmax_scale: float,
    out_data: torch.Tensor,
    out_scale: torch.Tensor,
    bf16_round: bool = False,
    compensate: bool | None = None,
    part_bf16: bool | None = None,
    vec: bool | None = None,
    direct: bool | None = None,
    pdl: bool | None = None,
    _parts: int = 3,
    _debug: int = 0,
    _bufs: tuple | None = None,
) -> None:
    """Drop-in for ``fused_norm_rope_attn_rope_cast_decode`` on <=N tokens.

    q: [s_q, HP, 512] bf16, chunk-interleaved, unroped (HP = padded heads).
    swa_cache / extra_cache: [num_blocks, page, bytes] uint8 views (528 / 288).
    Writes the live heads' groups of out_data [s_q, G, 4096] / out_scale.
    """
    assert _HAS_GLUON, "swap-AB attention needs Triton's Gluon Blackwell API"
    s_q, hp, dim = q.shape
    h = n_live_heads
    assert dim == _HEAD_DIM_V and h in (16, 32) and hp >= h
    assert q.dtype == torch.bfloat16 and q.stride(2) == 1 and q.stride(1) == dim
    assert swa_cache.shape[-1] == 528 and swa_cache.stride(-1) == 1
    assert swa_cache.stride(1) == swa_cache.shape[-1]
    assert swa_idx.stride(-1) == 1
    bt = _SWAPAB_BT
    nk = swa_idx.shape[-1]
    if extra_cache is None:
        ne = 0
        extra_cache, extra_idx, extra_len = swa_cache, swa_idx, swa_len
        e_fp4 = False
    else:
        assert extra_idx is not None and extra_len is not None
        assert extra_idx.stride(-1) == 1 and extra_cache.stride(-1) == 1
        assert extra_cache.shape[-1] in (528, 288)
        assert extra_cache.stride(1) == extra_cache.shape[-1]
        ne = extra_idx.shape[-1]
        e_fp4 = extra_cache.shape[-1] == 288
    kt = triton.cdiv(nk, bt)
    nt = kt + triton.cdiv(ne, bt)
    compensate = _SWAPAB_COMPENSATE if compensate is None else compensate
    part_bf16 = _SWAPAB_PART_BF16 if part_bf16 is None else part_bf16
    vec = _SWAPAB_VEC if vec is None else vec
    direct = _SWAPAB_DIRECT if direct is None else direct
    pdl = _SWAPAB_PDL if pdl is None else pdl
    if _bufs is not None:  # timing probes: reuse buffers across separate calls
        partial, maximum, sums = _bufs
    else:
        partial = torch.empty(
            (s_q, nt, h, dim),
            dtype=torch.bfloat16 if part_bf16 else torch.float32,
            device=q.device,
        )
        maximum = torch.empty((s_q, nt, h), dtype=torch.float32, device=q.device)
        sums = torch.empty_like(maximum)
    if _parts & 1:
      _at_partial[(s_q, nt)](
        q,
        swa_cache,
        extra_cache,
        swa_idx,
        extra_idx,
        swa_len,
        extra_len,
        positions_int32,
        cos_sin_cache,
        partial,
        maximum,
        sums,
        swa_cache.shape[0] * swa_cache.shape[1],
        extra_cache.shape[0] * extra_cache.shape[1],
        QS=q.stride(0),
        HP=hp,
        IS=swa_idx.stride(0),
        EIS=extra_idx.stride(0),
        CSS=cos_sin_cache.stride(0),
        KP=swa_cache.shape[1],
        KS=swa_cache.stride(0),
        EP=extra_cache.shape[1],
        ES=extra_cache.stride(0),
        NK=nk,
        NE=ne,
        NT=nt,
        KT=kt,
        BT=bt,
        H=h,
        SCALE=softmax_scale,
        E_FP4=e_fp4,
        COMPENSATE=compensate,
        DEBUG=_debug,
        VEC=vec,
        DIRECT=direct,
        num_warps=4,
    )
    bd = 64 if ne else 512
    if _parts & 2:
      _at_combine[(s_q, h, triton.cdiv(dim, bd))](
        partial,
        maximum,
        sums,
        sink,
        out_data,
        out_scale,
        cos_sin_cache,
        positions_int32,
        out_data.stride(0),
        out_data.stride(1),
        out_scale.stride(0),
        out_scale.stride(1),
        out_scale.stride(2),
        CSS=cos_sin_cache.stride(0),
        NT=nt,
        ST=triton.next_power_of_2(nt),
        H=h,
        BD=bd,
        GS=WV_GROUP_SIZE,
        EPS=1e-10,
        BF16_ROUND=bf16_round,
        USE_PDL=pdl,
        num_warps=4,
        launch_pdl=pdl,
    )


def dequant_mega_out(data: torch.Tensor, scale: torch.Tensor, n_heads: int) -> torch.Tensor:
    """[s_q, G, 4096] fp8 + packed ue8m0 scales (o-fused layout) -> [s_q, H, 512] fp32."""
    s_q = data.shape[0]
    g = n_heads // WV_GROUP_SIZE
    vals = data[:, :g].float().view(s_q, g, 128, 32)
    sb = scale[:, :g].contiguous().view(torch.uint8).view(s_q, g, 128).float()
    vals = vals * torch.exp2(sb - 127.0).unsqueeze(-1)
    # fused chunk k = c * 8 + hg -> [s_q, g, c, hg, j] -> [s_q, g, hg, c, j]
    vals = vals.view(s_q, g, 16, WV_GROUP_SIZE, 32).permute(0, 1, 3, 2, 4)
    return vals.reshape(s_q, g * WV_GROUP_SIZE, 512)


def is_flashmla_mega_attn_supported() -> tuple[bool, str | None]:
    """Whether FlashMLA's fused mega-attention kernels are usable here."""
    is_available, maybe_reason = is_flashmla_sparse_supported()
    if not is_available:
        return False, maybe_reason
    if not current_platform.is_device_capability_family(100):
        return False, "FlashMLA mega attention requires sm_10x GPUs."
    if not hasattr(torch.ops._flashmla_C, "fused_norm_rope_attn_rope_cast_decode"):
        return False, "vllm._flashmla_C was built without the mega attention ops."
    if not hasattr(torch.ops._flashmla_C, "permute_q_b_proj"):
        return False, "vllm._flashmla_C predates the mega attention weight permutes."
    return True, None


def alloc_mega_attn_output(
    num_tokens: int,
    n_wv_group: int,
    device: torch.device,
) -> QuantizedActivation:
    """Allocate the output buffer pair the mega-attention kernels write into.

    One pair per forward step: the prefill and decode segments fill disjoint
    token ranges, so a single ``fp8_einsum`` over all ``num_tokens`` can
    consume the result instead of one call per segment.

    The scale buffer is MN-major -- stride 1 along tokens, head-dim stride
    ``ceil4(num_tokens)`` -- which is both what the kernel requires of a
    caller-provided buffer and what DeepGEMM expects for this N.
    """
    d = WV_GROUP_SIZE * _HEAD_DIM_V
    data = torch.empty(
        (num_tokens, n_wv_group, d), dtype=torch.float8_e4m3fn, device=device
    )
    aligned = get_tma_aligned_size(num_tokens, torch.int32.itemsize)
    scale = torch.empty(
        (n_wv_group, d // (_QUANT_GROUP * 4), aligned),
        dtype=torch.int32,
        device=device,
    ).permute(2, 0, 1)[:num_tokens]
    return QuantizedActivation(
        data=data,
        scale=scale,
        orig_dtype=torch.bfloat16,
        orig_shape=data.shape,
        quant_key=kMxfp8Dynamic,
    )


class _PrefillChunk(NamedTuple):
    """One prefill chunk's step-constant inputs, shared by all layers of a type."""

    size: int  # requests
    n: int  # compressed width
    m: int  # gathered width (compressed + SWA)
    qs: int  # query rows [qs, qe), relative to the first prefill row
    qe: int
    seq_lens: torch.Tensor
    gather_lens: torch.Tensor
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    compressed_seq_lens: torch.Tensor | None
    compressed_block_table: torch.Tensor | None


def _token_slice(out: QuantizedActivation, start: int, end: int) -> QuantizedActivation:
    """The ``[start, end)`` token slice of a mega-attention output buffer."""
    data = out.data[start:end]
    return replace(out, data=data, scale=out.scale[start:end], orig_shape=data.shape)


class DeepseekV4MegaAttnAttention(DeepseekV4FlashMLAAttention):
    """FlashMLA mega-attention layer for DeepSeek V4.1 (SM100)."""

    backend_cls = FlashMLAMegaAttnBackend

    @classmethod
    def is_available_for(cls, vllm_config: VllmConfig) -> bool:
        """Whether this layer can serve the configured model on this device.

        The kernel is SM100-only and wants ``WV_GROUP_SIZE`` heads per ``wo_a``
        group, which stops holding once TP divides the head count far enough
        (TP16 on a 64-head model leaves 4). ``__init__`` raises on both, so the
        default-backend selector asks here first rather than turning an
        unsupported topology into a startup crash.
        """
        if not is_flashmla_mega_attn_supported()[0]:
            return False
        config = vllm_config.model_config.hf_config
        config = getattr(config, "text_config", config)
        n_heads = getattr(config, "num_attention_heads", 0) or 0
        n_groups = getattr(config, "o_groups", 0) or 0
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        if n_heads % tp_size or n_groups % tp_size:
            return False
        n_local_heads, n_local_groups = n_heads // tp_size, n_groups // tp_size
        if not n_local_groups or n_local_heads % WV_GROUP_SIZE:
            return False
        return n_local_heads // n_local_groups == WV_GROUP_SIZE

    def __init__(self, vllm_config: VllmConfig, *args, **kwargs) -> None:
        super().__init__(vllm_config, *args, **kwargs)
        if self.n_local_heads % WV_GROUP_SIZE:
            raise ValueError(
                f"{self.prefix}: mega attention needs the local head count "
                f"({self.n_local_heads}) to be a multiple of {WV_GROUP_SIZE}."
            )
        if self.n_local_heads // self.n_local_groups != WV_GROUP_SIZE:
            raise ValueError(
                f"{self.prefix}: mega attention needs {WV_GROUP_SIZE} heads per "
                "wo_a group."
            )
        self.n_wv_group = self.padded_heads // WV_GROUP_SIZE
        self._fused_layouts_ready = False

    # ---- interface contract ------------------------------------------------

    @property
    def accepts_unnormed_unroped_query(self) -> bool:
        return True

    @property
    def packed_kv_cache_dtype(self) -> CacheDType:
        # This kernel is the only one that reads an NVFP4 compressed cache, so
        # it is what an unspecific --kv-cache-dtype resolves to here.
        return "nvfp4_ds_mla"

    def _alloc_attn_out(
        self, num_tokens: int, hidden_states: torch.Tensor
    ) -> QuantizedActivation:
        return alloc_mega_attn_output(num_tokens, self.n_wv_group, hidden_states.device)

    def _o_proj(
        self, attn_out: QuantizedActivation, positions: torch.Tensor
    ) -> torch.Tensor:
        """Grouped ``wo_a`` then ``wo_b`` over the kernel's quantized output.

        The inverse RoPE and the FP8 cast happened inside the attention
        kernel, and ``wo_a`` is permuted for its output layout, so there is
        nothing to rotate or quantize here and ``positions`` is unused. The
        scale already is DeepGEMM's packed-ue8m0 MN-major layout, so the
        einsum consumes the kernel's output with no repacking.
        """
        groups = self.n_local_groups
        z = torch.empty(
            (attn_out.data.shape[0], groups, self.o_lora_rank),
            dtype=torch.bfloat16,
            device=attn_out.data.device,
        )
        fp8_einsum(
            "bhr,hdr->bhd",
            (attn_out.data[:, :groups], attn_out.scale[:, :groups]),
            (self.wo_a.weight, self.wo_a.weight_scale),
            z,
            recipe=self._einsum_recipe,
        )
        return self._wo_b_proj(z.flatten(1))

    # ---- weights -----------------------------------------------------------

    def finalize_loaded_weights(self) -> None:
        """Permute wq_b rows / wo_a columns into the kernel's layouts.

        Idempotent: a second post-load pass must not permute twice.
        """
        if self._fused_layouts_ready:
            return
        permute_wq_b_(
            self.wq_b.weight.data, self.wq_b.weight_scale.data, self.n_local_heads
        )
        permute_wo_a_(
            self.wo_a.weight.data,
            self.wo_a.weight_scale.data,
            self.n_local_heads // self.n_local_groups,
        )
        self._fused_layouts_ready = True

    # ---- forward -----------------------------------------------------------

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: QuantizedActivation,
    ) -> None:
        if not self._fused_layouts_ready:
            raise RuntimeError(
                f"{self.prefix}: wq_b / wo_a were never permuted for the mega "
                "attention kernel; refusing to run with mismatched layouts."
            )
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None:
            # Warmup dummy run: reserve the prefill workspace, produce zeros.
            self._reserve_prefill_workspace(q)
            output.data.zero_()
            output.scale.zero_()
            return

        assert isinstance(attn_metadata, dict)
        # q was zero-padded to the kernel's head count by the fused KV insert.
        assert q.shape[1] == self.padded_heads
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None,
            attn_metadata.get(self.compressed_cache_prefix)
            if self.compressed_cache_prefix is not None
            else None,
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata", attn_metadata[self.swa_cache_layer.prefix]
        )
        # The kernel takes int32 RoPE positions.
        positions_int32 = positions.to(torch.int32)
        num_decode_tokens = swa_metadata.num_decode_tokens

        if swa_metadata.num_prefills > 0:
            self._forward_prefill_mega(
                q[num_decode_tokens:],
                positions_int32[num_decode_tokens:],
                flashmla_metadata,
                swa_metadata,
                output,
                num_decode_tokens,
            )
        if swa_metadata.num_decodes > 0:
            self._forward_decode_mega(
                q[:num_decode_tokens],
                positions_int32[:num_decode_tokens],
                flashmla_metadata,
                swa_metadata,
                _token_slice(output, 0, num_decode_tokens),
            )

    def _reserve_prefill_workspace(self, q: torch.Tensor) -> None:
        swa_only = self.compress_ratio == 0
        n = 0 if swa_only else -(-self.max_model_len // self.compress_ratio)
        m = n + self.window_size + self.max_num_batched_tokens
        if swa_only:
            top_k = 0
        else:
            assert self.topk_indices_buffer is not None
            top_k = self.topk_indices_buffer.shape[-1]
        combined_topk = round_up(top_k + self.window_size, 128)
        current_workspace_manager().get_simultaneous(
            ((self.PREFILL_CHUNK_SIZE, m, q.shape[-1]), torch.bfloat16),
            ((self.max_num_batched_tokens, combined_topk), torch.int32),
            ((self.max_num_batched_tokens,), torch.int32),
        )

    def _decode_compressed_kv_and_topk(
        self,
        flashmla_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """The compressed cache and its ``[n, topk]`` slot indices / lengths."""
        if self.compress_ratio == 0:
            return None, None, None
        assert flashmla_metadata is not None
        assert swa_metadata.is_valid_token is not None
        assert self.topk_indices_buffer is not None
        num_decode_tokens = swa_metadata.num_decode_tokens
        indices, lens = compute_global_topk_indices_and_lens(
            self.topk_indices_buffer[:num_decode_tokens],
            swa_metadata.token_to_req_indices,
            flashmla_metadata.block_table[: swa_metadata.num_decodes],
            flashmla_metadata.block_size // self.compress_ratio,
            swa_metadata.is_valid_token[:num_decode_tokens],
        )
        return self._compressed_kv_cache().unsqueeze(-2), indices, lens

    def _forward_decode_mega(
        self,
        q: torch.Tensor,
        positions_int32: torch.Tensor,
        flashmla_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        out: QuantizedActivation,
    ) -> None:
        """Mega attention over the paged quantized caches, in place into ``out``.

        The caches are ``[num_blocks, page, 1, bytes]``, the record taken from
        ``bytes`` (528 V4.1 fp8, 288 V4.1 NVFP4; NVFP4 only as the compressed
        cache beside an fp8 SWA one), and the indices ``[s_q, topk]`` int32
        slot ids (``block * page + offset``, ``-1`` invalid).
        """
        extra_cache, extra_idx, extra_len = self._decode_compressed_kv_and_topk(
            flashmla_metadata, swa_metadata
        )
        assert swa_metadata.decode_swa_indices is not None
        use_swapab = (
            _SWAPAB_MODE in ("1", "2")
            and q.shape[0] <= _SWAPAB_MAX_TOKENS
            and swapab_supported(self.n_local_heads)
            and (extra_cache is not None or _SWAPAB_SWA_ONLY)
        )
        if use_swapab and _SWAPAB_MODE == "1":
            swapab_mega_decode(
                q,
                self.swa_cache_layer.kv_cache,
                swa_metadata.decode_swa_indices.view(q.shape[0], -1),
                swa_metadata.decode_swa_lens,
                None if extra_cache is None else extra_cache.squeeze(-2),
                extra_idx,
                extra_len,
                positions_int32,
                self.rotary_emb.cos_sin_cache,
                self.attn_sink,
                self.n_local_heads,
                self.scale,
                out.data,
                out.scale,
            )
            return
        torch.ops._flashmla_C.fused_norm_rope_attn_rope_cast_decode(
            q,
            self.swa_cache_layer.kv_cache.unsqueeze(-2),
            swa_metadata.decode_swa_indices.view(q.shape[0], -1),
            self.scale,
            _HEAD_DIM_V,
            self.attn_sink,
            swa_metadata.decode_swa_lens,
            extra_cache,
            extra_idx,
            extra_len,
            False,  # enable_q_norm
            0.0,  # rms_norm_eps
            positions_int32,
            *_ROPE_ARGS,
            self.rotary_emb.cos_sin_cache,
            self.n_wv_group,
            *_SF_ARGS,
            out.data,
            out.scale,
        )
        if use_swapab and not torch.cuda.is_current_stream_capturing():
            self._swapab_shadow(
                q,
                swa_metadata,
                extra_cache,
                extra_idx,
                extra_len,
                positions_int32,
                out,
            )

    def _swapab_shadow(
        self, q, swa_metadata, extra_cache, extra_idx, extra_len, positions_int32, out
    ) -> None:
        """Debug mode 2: run swap-AB beside the mega kernel and log the diff."""
        global _swapab_shadow_logged
        if _swapab_shadow_logged >= _SWAPAB_SHADOW_LOG_LIMIT:
            return
        tmp = alloc_mega_attn_output(q.shape[0], self.n_wv_group, q.device)
        swapab_mega_decode(
            q,
            self.swa_cache_layer.kv_cache,
            swa_metadata.decode_swa_indices.view(q.shape[0], -1),
            swa_metadata.decode_swa_lens,
            None if extra_cache is None else extra_cache.squeeze(-2),
            extra_idx,
            extra_len,
            positions_int32,
            self.rotary_emb.cos_sin_cache,
            self.attn_sink,
            self.n_local_heads,
            self.scale,
            tmp.data,
            tmp.scale,
        )
        ref = dequant_mega_out(out.data, out.scale, self.n_local_heads)
        got = dequant_mega_out(tmp.data, tmp.scale, self.n_local_heads)
        diff = (ref - got).abs()
        amax = ref.abs().amax().clamp_min(1e-30)
        rel = (diff.norm() / ref.norm().clamp_min(1e-30)).item()
        msg = (
            f"[swapab-shadow] {self.prefix} cr={self.compress_ratio} s_q={q.shape[0]} "
            f"max_abs={diff.amax().item():.4e} ref_amax={amax.item():.4e} "
            f"max_rel={(diff.amax() / amax).item():.4e} fro_rel={rel:.4e} "
            f"swa_len={swa_metadata.decode_swa_lens[: q.shape[0]].tolist()} "
            f"extra_len={None if extra_len is None else extra_len[: q.shape[0]].tolist()}"
        )
        print(msg, flush=True)
        _swapab_shadow_logged += 1

    def _prefill_chunks(
        self,
        flashmla_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> list[_PrefillChunk]:
        """Slices and lengths of each prefill chunk, computed once per step.

        They depend only on the step's metadata and this layer type's compress
        ratio, so every layer of the type reuses them (values are identical to
        recomputing them per layer).
        """
        swa_only = self.compress_ratio == 0
        key = (
            "mega_prefill_chunks",
            self.compress_ratio,
            self.PREFILL_CHUNK_SIZE,
            None if swa_only else id(flashmla_metadata),
        )
        chunks = swa_metadata.prefill_chunk_cache.get(key)
        if chunks is not None:
            return chunks
        num_decodes = swa_metadata.num_decodes
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert seq_lens is not None and gather_lens is not None
        assert query_start_loc_cpu is not None and query_start_loc is not None
        chunk_plan = swa_metadata.get_prefill_chunk_plan(
            compress_ratio=self.compress_ratio,
            prefill_chunk_size=self.PREFILL_CHUNK_SIZE,
            has_compressed=not swa_only,
        )
        assert chunk_plan, "prefill chunk plan must be non-empty when num_prefills > 0"
        qsl = query_start_loc_cpu.tolist()
        prefill_token_base = qsl[num_decodes]
        swa_block_table = swa_metadata.block_table[num_decodes:]
        compressed_block_table = None
        if not swa_only:
            assert flashmla_metadata is not None
            compressed_block_table = flashmla_metadata.block_table[num_decodes:]
        chunks = []
        for chunk_start, chunk_end, chunk_n, chunk_m in chunk_plan:
            first, last = num_decodes + chunk_start, num_decodes + chunk_end
            chunk_seq_lens = seq_lens[chunk_start:chunk_end]
            chunks.append(
                _PrefillChunk(
                    size=chunk_end - chunk_start,
                    n=chunk_n,
                    m=chunk_m,
                    qs=qsl[first] - prefill_token_base,
                    qe=qsl[last] - prefill_token_base,
                    seq_lens=chunk_seq_lens,
                    gather_lens=gather_lens[chunk_start:chunk_end],
                    block_table=swa_block_table[chunk_start:chunk_end],
                    query_start_loc=query_start_loc[first : last + 1],
                    compressed_seq_lens=(
                        None if swa_only else chunk_seq_lens // self.compress_ratio
                    ),
                    compressed_block_table=(
                        None
                        if compressed_block_table is None
                        else compressed_block_table[chunk_start:chunk_end]
                    ),
                )
            )
        swa_metadata.prefill_chunk_cache[key] = chunks
        return chunks

    def _forward_prefill_mega(
        self,
        q: torch.Tensor,
        positions_int32: torch.Tensor,
        flashmla_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        out: QuantizedActivation,
        token_base: int,
    ) -> None:
        swa_only = self.compress_ratio == 0
        num_decode_tokens = swa_metadata.num_decode_tokens
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[num_decode_tokens:][
            : swa_metadata.num_prefill_tokens
        ]
        top_k = 0 if swa_only else topk_indices.shape[-1]
        workspace_manager = current_workspace_manager()
        combined_topk = round_up(top_k + self.window_size, 128)
        for chunk in self._prefill_chunks(flashmla_metadata, swa_metadata):
            kv_ws, idx_ws, lens_ws = workspace_manager.get_simultaneous(
                ((chunk.size, chunk.m, q.shape[-1]), torch.bfloat16),
                ((self.max_num_batched_tokens, combined_topk), torch.int32),
                ((self.max_num_batched_tokens,), torch.int32),
            )
            if not swa_only:
                assert flashmla_metadata is not None
                dequantize_and_gather_k_cache(
                    kv_ws[: chunk.size, : chunk.n],
                    self._compressed_kv_cache(),
                    seq_lens=chunk.compressed_seq_lens,
                    gather_lens=None,
                    block_table=chunk.compressed_block_table,
                    block_size=flashmla_metadata.block_size // self.compress_ratio,
                    offset=0,
                )
            dequantize_and_gather_k_cache(
                kv_ws[: chunk.size],
                self.swa_cache_layer.kv_cache,
                seq_lens=chunk.seq_lens,
                gather_lens=chunk.gather_lens,
                block_table=chunk.block_table,
                block_size=swa_metadata.block_size,
                offset=chunk.n,
            )
            qs, qe = chunk.qs, chunk.qe
            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[qs:qe],
                chunk.query_start_loc,
                chunk.seq_lens,
                chunk.gather_lens,
                self.window_size,
                self.compress_ratio,
                top_k,
                chunk.m,
                chunk.n,
                out=(idx_ws[: qe - qs], lens_ws[: qe - qs]),
            )
            chunk_out = _token_slice(out, token_base + qs, token_base + qe)
            # Mega attention over the gathered non-paged bf16 KV (RoPE already
            # applied), in place into this chunk's token range. `indices`
            # entries outside [0, s_kv) skip.
            torch.ops._flashmla_C.fused_norm_rope_attn_rope_cast_fwd(
                q[qs:qe],
                kv_ws.view(-1, 1, q.shape[-1]),
                combined_indices.unsqueeze(1),
                self.scale,
                _HEAD_DIM_V,
                self.attn_sink,
                combined_lens,
                False,  # enable_q_norm
                0.0,  # rms_norm_eps
                positions_int32[qs:qe],
                *_ROPE_ARGS,
                self.rotary_emb.cos_sin_cache,
                self.n_wv_group,
                *_SF_ARGS,
                chunk_out.data,
                chunk_out.scale,
            )
