# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch

from vllm.triton_utils import tl, triton

# Opt-in exact replacement for ``scores.topk(k)`` + ``_store_candidates_kernel``
# in select_candidate_blocks (see _candidate_topk_kernel).
FAST_CANDIDATE_TOPK = os.environ.get("VLLM_DSV41_FAST_CANDIDATE_TOPK", "0") == "1"
# kernel-agent patch (10/12), opt-in VLLM_KN_BOUNDED_CANDIDATE_MASK=1: apply_candidate_mask only rewrites each row's
# [start, min(end, width)) columns -- the only ones the downstream top-k kernels read (topKPerRowPrefill / decode top-k
# take per-row bounds; DeepGEMM logits are produced with clean_logits=False, so columns past a row's end are garbage
# anyway). Values inside the range are bit-identical to the original sweep, which also wrote -inf over the whole
# (max_model_len // ratio)-wide row. Graph-safe: fixed grid, per-row dynamic tile loop.
BOUNDED_CANDIDATE_MASK = os.environ.get("VLLM_KN_BOUNDED_CANDIDATE_MASK", "0") == "1"
_BOUNDED_MASK_PROGS = int(os.environ.get("VLLM_KN_BOUNDED_MASK_PROGS", "8"))


@triton.jit
def _max_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit(do_not_specialize=["width", "nblocks"])
def _block_scores_kernel(
    logits,
    starts,
    ends,
    scores,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
    SKIP_TAIL: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    blocks = tl.program_id(1) * TILE + tl.arange(0, TILE)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    if SKIP_TAIL:
        # The bounded fast top-k never reads blocks past the row's extent (the
        # same bound as in _candidate_topk_kernel), so tiles entirely past it
        # skip their -inf stores.
        lim = tl.minimum(end, width)
        live = tl.where(lim > start, (lim - start + BLOCK_SIZE - 1) // BLOCK_SIZE, 0)
        pin = tl.where(end > start, (end - start - 1) // BLOCK_SIZE + 1, 0)
        if tl.program_id(1) * TILE >= tl.maximum(live, pin):
            return
    offsets = tl.arange(0, triton.next_power_of_2(BLOCK_SIZE))
    cols = start + blocks[:, None] * BLOCK_SIZE + offsets[None, :]
    values = tl.load(
        logits + row * stride_row + cols * stride_col,
        (blocks[:, None] < nblocks)
        & (offsets[None, :] < BLOCK_SIZE)
        & (cols < end)
        & (cols < width),
        other=-float("inf"),
    )
    reduced = tl.reduce(values, 1, _max_with_nan)
    reduced = tl.where(
        (end > start) & (blocks == (end - start - 1) // BLOCK_SIZE),
        float("inf"),
        reduced,
    )
    tl.store(scores + row * nblocks + blocks, reduced, blocks < nblocks)


@triton.jit(do_not_specialize=["k"])
def _store_candidates_kernel(
    values,
    indices,
    output,
    out_stride_row,
    out_stride_col,
    k,
    OUT_K: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    value = tl.load(values + row * k + cols, cols < k, other=-float("inf"))
    index = tl.load(indices + row * k + cols, cols < k, other=-1)
    # NaN scores can occur during warmup; only -inf denotes padding.
    tl.store(
        output + row * out_stride_row + cols * out_stride_col,
        tl.where(value != -float("inf"), index, -1),
        cols < OUT_K,
    )


@triton.jit
def _topk_select_key(v):
    """torch.topk's radix key (TopKTypeConfig<float>::convert): IEEE order on
    unsigned ints, every NaN ranked highest."""
    bits = v.to(tl.uint32, bitcast=True)
    key = bits ^ tl.where((bits >> 31) != 0, 0xFFFFFFFF, 0x80000000)
    return tl.where(v != v, 0xFFFFFFFF, key)


@triton.jit
def _topk_sort_key(v, NEG_ZERO_EQ: tl.constexpr):
    """Key of the CUB radix sort torch.topk(sorted=True) applies to the selected
    values (stable, descending); CUB ranks -0.0 equal to +0.0."""
    bits = v.to(tl.uint32, bitcast=True)
    if NEG_ZERO_EQ:
        bits = tl.where(bits == 0x80000000, tl.zeros_like(bits), bits)
    return bits ^ tl.where((bits >> 31) != 0, 0xFFFFFFFF, 0x80000000)


@triton.jit
def _topk_digit_hist(keys, ok, prefix, p: tl.constexpr, n_tail, bins):
    """256-bin histogram (in 512 bins, upper half ignored) of byte p (MSB first)
    of the select keys whose higher bytes equal ``prefix``, plus ``n_tail``
    implied -inf keys (0x007FFFFF)."""
    if p > 0:
        ok = ok & ((keys >> (32 - 8 * p)) == (prefix >> (32 - 8 * p)))
    digit = ((keys >> (24 - 8 * p)) & 0xFF).to(tl.int32)
    return tl.histogram(tl.where(ok, digit, 256), 512)


@triton.jit
def _topk_pick_digit(hist, remaining, prefix, p: tl.constexpr, n_tail, bins):
    """Pick byte p of the k-th largest key; returns (prefix, remaining, number of
    read (non-tail) keys sharing the new prefix)."""
    # The unread tail: n_tail blocks with the select key of -inf (0x007FFFFF).
    tail_digit = (0x007FFFFF >> (24 - 8 * p)) & 0xFF
    read_hist = hist
    if p == 0:
        hist += tl.where(bins == tail_digit, n_tail, 0)
    else:
        tail_in = (prefix >> (32 - 8 * p)) == (0x007FFFFF >> (32 - 8 * p))
        hist += tl.where((bins == tail_digit) & tail_in, n_tail, 0)
    h = tl.where(bins < 256, hist, 0)
    gt = tl.sum(h, 0) - tl.cumsum(h, 0)  # count of digits > b
    d = tl.max(tl.where((gt + h >= remaining) & (bins < 256), bins, -1), 0)
    remaining = remaining - tl.sum(tl.where(bins == d, gt, 0), 0)
    n_read = tl.sum(tl.where(bins == d, read_hist, 0), 0)
    if p == 0:
        prefix = d.to(tl.uint32) << (24 - 8 * p)
    else:
        prefix = prefix | (d.to(tl.uint32) << (24 - 8 * p))
    return prefix, remaining, n_read


@triton.jit
def _topk_pack(v, gt, cols, NEG_ZERO_EQ: tl.constexpr):
    """Sort record: CUB sort key, then torch's gather order (keys > T before the
    kept ties, each in index order), then the index (21 bits, inverted)."""
    return (
        (_topk_sort_key(v, NEG_ZERO_EQ).to(tl.uint64) << 32)
        | (gt.to(tl.uint64) << 31)
        | (0x1FFFFF - cols).to(tl.uint64)
    )


@triton.jit(do_not_specialize=["nblocks", "k", "width"])
def _candidate_topk_kernel(
    scores,
    scratch,
    cand,
    meta,
    out,
    out_stride_row,
    out_stride_col,
    nblocks,
    k,
    starts,
    ends,
    stride_start,
    stride_end,
    width,
    OUT_K: tl.constexpr,
    KPAD: tl.constexpr,
    CHUNK: tl.constexpr,
    CAP: tl.constexpr,
    NEG_ZERO_EQ: tl.constexpr,
    BOUNDED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    SPLIT: tl.constexpr,
):
    """One row of ``top = scores.topk(k); _store_candidates_kernel(top)``.

    Bit-exact with torch.topk(largest=True, sorted=True):
      1. Radix select (4 x 8-bit histograms, MSB first) finds the k-th largest
         select-key T and how many T-ties to keep (the lowest indices, as
         torch's gatherTopK keeps the first-seen ties).
      2. Gather the selected entries (keys > T, then the kept ties).
      3. Stable descending sort by the CUB sort key, ties in torch's gather
         order (keys > T first, then index order), via one bitonic sort of
         packed records; -inf -> -1 as in _store_candidates_kernel.
    With CAP > 0 only the top-byte histogram (and, if T's top-byte bucket holds
    more than CAP keys, the second-byte one) reads the whole row; one more full
    pass writes the entries above T's prefix to the output scratch and compacts
    the ones sharing it into ``cand``; the low-byte passes run on ``cand``. If
    even the 16-bit bucket overflows CAP (massive ties), all passes read the row.
    With BOUNDED, blocks past the row's extent (which _block_scores_kernel
    fills with -inf) are not read: they are counted into the histograms as
    -inf and, when selected, synthesized as -inf entries in index order.
    """
    row = tl.program_id(0).to(tl.int64)
    src = scores + row * nblocks
    dst = scratch + row * KPAD
    cdst = cand + row * CAP
    bins = tl.arange(0, 512)
    remaining = k
    if BOUNDED:
        bound = row // ROW_REPEAT
        start = tl.load(starts + bound * stride_start) if HAS_STARTS else 0
        end = tl.load(ends + bound * stride_end)
        lim = tl.minimum(end, width)
        live = tl.where(lim > start, (lim - start + BLOCK_SIZE - 1) // BLOCK_SIZE, 0)
        # _block_scores_kernel pins block (end - start - 1) // BLOCK_SIZE to +inf.
        pin = tl.where(end > start, (end - start - 1) // BLOCK_SIZE + 1, 0)
        # int32 like nblocks: the bounds may be int64 tensors.
        n_scan = tl.minimum(tl.maximum(live, pin), nblocks).to(tl.int32)
    else:
        n_scan = nblocks
    n_tail = nblocks - n_scan

    # Pass 0 (full row): top byte.
    hist = tl.zeros((512,), dtype=tl.int32)
    for c0 in range(0, n_scan, CHUNK):
        cols = c0 + tl.arange(0, CHUNK)
        inb = cols < n_scan
        keys = _topk_select_key(tl.load(src + cols, inb, other=0.0))
        hist += _topk_digit_hist(keys, inb, 0, 0, n_tail, bins)
    prefix, remaining, n_cand = _topk_pick_digit(hist, remaining, 0, 0, n_tail, bins)
    # Second full-row pass only if T's top-byte bucket is too big to compact.
    two = n_cand > CAP
    if two:
        hist = tl.zeros((512,), dtype=tl.int32)
        for c0 in range(0, n_scan, CHUNK):
            cols = c0 + tl.arange(0, CHUNK)
            inb = cols < n_scan
            keys = _topk_select_key(tl.load(src + cols, inb, other=0.0))
            hist += _topk_digit_hist(keys, inb, prefix, 1, n_tail, bins)
        prefix, remaining, n_cand = _topk_pick_digit(
            hist, remaining, prefix, 1, n_tail, bins
        )
    shift = tl.where(two, 16, 24).to(tl.uint32)
    use_cand = (n_cand <= CAP) & (CAP > 0)

    if use_cand:
        # Full row once more: entries above T's prefix are final (dst, index
        # order); the ones sharing it are compacted into cand (index order).
        base_gt = k * 0
        base_c = k * 0
        for c0 in range(0, n_scan, CHUNK):
            cols = c0 + tl.arange(0, CHUNK)
            inb = cols < n_scan
            v = tl.load(src + cols, inb, other=0.0)
            keys = _topk_select_key(v)
            hi = inb & ((keys >> shift) > (prefix >> shift))
            eq = inb & ((keys >> shift) == (prefix >> shift))
            gi = hi.to(tl.int32)
            ei = eq.to(tl.int32)
            r_gt = base_gt + tl.cumsum(gi, 0) - gi
            r_c = base_c + tl.cumsum(ei, 0) - ei
            tl.store(
                dst + r_gt,
                _topk_pack(v, gi, cols, NEG_ZERO_EQ).to(tl.int64, bitcast=True),
                hi,
            )
            # Raw score bits (NaN payloads matter to the sort key) + index.
            rec = (v.to(tl.uint32, bitcast=True).to(tl.uint64) << 32) | cols.to(
                tl.uint64
            )
            tl.store(cdst + r_c, rec.to(tl.int64, bitcast=True), eq)
            base_gt += tl.sum(gi, 0)
            base_c += tl.sum(ei, 0)
        # The compacted candidates are read back by other threads.
        tl.debug_barrier()
        for p in tl.static_range(1, 4):
            if (p > 1) | (not two):
                hist = tl.zeros((512,), dtype=tl.int32)
                for c0 in range(0, n_cand, CHUNK):
                    lanes = c0 + tl.arange(0, CHUNK)
                    inb = lanes < n_cand
                    rec = tl.load(cdst + lanes, inb, other=0).to(
                        tl.uint64, bitcast=True
                    )
                    v = (rec >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
                    hist += _topk_digit_hist(
                        _topk_select_key(v), inb, prefix, p, n_tail, bins
                    )
                prefix, remaining, _ = _topk_pick_digit(
                    hist, remaining, prefix, p, n_tail, bins
                )
        n_gt = k - remaining
        base_eq = k * 0
        for c0 in range(0, n_cand, CHUNK):
            lanes = c0 + tl.arange(0, CHUNK)
            inb = lanes < n_cand
            rec = tl.load(cdst + lanes, inb, other=0).to(tl.uint64, bitcast=True)
            v = (rec >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
            keys = _topk_select_key(v)
            cols = (rec & 0x1FFFFF).to(tl.int32)
            is_gt = inb & (keys > prefix)
            is_eq = inb & (keys == prefix)
            gi = is_gt.to(tl.int32)
            ei = is_eq.to(tl.int32)
            r_gt = base_gt + tl.cumsum(gi, 0) - gi
            r_eq = base_eq + tl.cumsum(ei, 0) - ei
            take = is_gt | (is_eq & (r_eq < remaining))
            slot = tl.where(is_gt, r_gt, n_gt + r_eq)
            tl.store(
                dst + slot,
                _topk_pack(v, gi, cols, NEG_ZERO_EQ).to(tl.int64, bitcast=True),
                take,
            )
            base_gt += tl.sum(gi, 0)
            base_eq += tl.sum(ei, 0)
    else:
        for p in tl.static_range(1, 4):
            if (p > 1) | (not two):
                hist = tl.zeros((512,), dtype=tl.int32)
                for c0 in range(0, n_scan, CHUNK):
                    cols = c0 + tl.arange(0, CHUNK)
                    inb = cols < n_scan
                    keys = _topk_select_key(tl.load(src + cols, inb, other=0.0))
                    hist += _topk_digit_hist(keys, inb, prefix, p, n_tail, bins)
                prefix, remaining, _ = _topk_pick_digit(
                    hist, remaining, prefix, p, n_tail, bins
                )
        n_gt = k - remaining
        base_gt = k * 0
        base_eq = k * 0
        for c0 in range(0, n_scan, CHUNK):
            cols = c0 + tl.arange(0, CHUNK)
            inb = cols < n_scan
            v = tl.load(src + cols, inb, other=0.0)
            keys = _topk_select_key(v)
            is_gt = inb & (keys > prefix)
            is_eq = inb & (keys == prefix)
            gi = is_gt.to(tl.int32)
            ei = is_eq.to(tl.int32)
            r_gt = base_gt + tl.cumsum(gi, 0) - gi
            r_eq = base_eq + tl.cumsum(ei, 0) - ei
            take = is_gt | (is_eq & (r_eq < remaining))
            slot = tl.where(is_gt, r_gt, n_gt + r_eq)
            tl.store(
                dst + slot,
                _topk_pack(v, gi, cols, NEG_ZERO_EQ).to(tl.int64, bitcast=True),
                take,
            )
            base_gt += tl.sum(gi, 0)
            base_eq += tl.sum(ei, 0)
    filled = n_gt + tl.minimum(base_eq, remaining)
    if SPLIT:
        # Sort in _candidate_sort_kernel (higher occupancy than this kernel).
        tl.store(meta + row * 2, filled)
        tl.store(meta + row * 2 + 1, n_scan)
    else:
        # The sort reads slots other threads wrote.
        tl.debug_barrier()
        _topk_sort_store(
            dst, filled, n_scan, k, out, row, out_stride_row, out_stride_col,
            OUT_K, KPAD,
        )


@triton.jit
def _topk_sort_store(
    dst, filled, n_scan, k, out, row, out_stride_row, out_stride_col,
    OUT_K: tl.constexpr, KPAD: tl.constexpr,
):
    lanes = tl.arange(0, KPAD)
    packed = tl.load(dst + lanes, lanes < filled, other=0).to(tl.uint64, bitcast=True)
    # Selected unread tail blocks (only when T is -inf): ties after the read ones.
    tail = (tl.full((KPAD,), 0x007FFFFF, tl.uint64) << 32) | (
        0x1FFFFF - (n_scan + lanes - filled)
    ).to(tl.uint64)
    packed = tl.where((lanes >= filled) & (lanes < k), tail, packed)
    packed = tl.sort(packed, descending=True)
    index = (0x1FFFFF - (packed & 0x1FFFFF)).to(tl.int32)
    # Sort key of -inf; _store_candidates_kernel maps -inf scores to -1.
    finite = (packed >> 32).to(tl.uint32) != 0x007FFFFF
    tl.store(
        out + row * out_stride_row + lanes * out_stride_col,
        tl.where((lanes < k) & finite, index, -1),
        lanes < OUT_K,
    )


@triton.jit(do_not_specialize=["k"])
def _candidate_sort_kernel(
    scratch,
    meta,
    out,
    out_stride_row,
    out_stride_col,
    k,
    OUT_K: tl.constexpr,
    KPAD: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    _topk_sort_store(
        scratch + row * KPAD,
        tl.load(meta + row * 2),
        tl.load(meta + row * 2 + 1),
        k,
        out,
        row,
        out_stride_row,
        out_stride_col,
        OUT_K,
        KPAD,
    )


# CUB's radix sort ranks -0.0 equal to +0.0 (verified against torch.topk on
# device by kernels/dense/bench/bench_cand_topk.py).
_TOPK_NEG_ZERO_EQ = True
# Candidates sharing the k-th key's top byte that are compacted for the low-byte
# passes (0 = always run the four histogram passes over the whole row).
_TOPK_CAP = int(os.environ.get("VLLM_DSV41_CANDIDATE_TOPK_CAP", "4096"))
_TOPK_WARPS = int(os.environ.get("VLLM_DSV41_CANDIDATE_TOPK_WARPS", "8"))
# Sort in a second kernel (selection and sort have different occupancy limits).
_TOPK_SPLIT = os.environ.get("VLLM_DSV41_CANDIDATE_TOPK_SPLIT", "1") == "1"
_TOPK_SORT_WARPS = int(os.environ.get("VLLM_DSV41_CANDIDATE_TOPK_SORT_WARPS", "8"))


def fast_select_topk(
    scores: torch.Tensor,
    k: int,
    out: torch.Tensor,
    row_ks: torch.Tensor | None = None,
    row_ke: torch.Tensor | None = None,
    width: int = 0,
    block_size: int = 1,
    row_repeat: int = 1,
) -> None:
    """``top = scores.topk(k)`` + ``_store_candidates_kernel`` in one kernel.

    With ``row_ke`` (and the _block_scores_kernel arguments that produced
    ``scores``), each row only reads its live blocks; the -inf tail is implied.
    """
    rows, nblocks = scores.shape
    out_k = out.shape[1]
    kpad = triton.next_power_of_2(out_k)
    assert scores.is_contiguous() and scores.dtype == torch.float32
    assert 0 < k <= min(out_k, nblocks) and kpad <= 2048 and nblocks < (1 << 21) - 1
    scratch = torch.empty((rows, kpad), dtype=torch.int64, device=scores.device)
    cap = _TOPK_CAP if nblocks > _TOPK_CAP else 0
    cand = torch.empty((rows, max(cap, 1)), dtype=torch.int64, device=scores.device)
    meta = torch.empty((rows, 2), dtype=torch.int32, device=scores.device)
    bounded = row_ke is not None
    _candidate_topk_kernel[(rows,)](
        scores,
        scratch,
        cand,
        meta,
        out,
        *out.stride(),
        nblocks,
        k,
        row_ks,
        row_ke,
        row_ks.stride(0) if row_ks is not None else 0,
        row_ke.stride(0) if bounded else 0,
        width,
        OUT_K=out_k,
        KPAD=kpad,
        CHUNK=4096,
        CAP=cap,
        NEG_ZERO_EQ=_TOPK_NEG_ZERO_EQ,
        BOUNDED=bounded,
        BLOCK_SIZE=block_size,
        HAS_STARTS=row_ks is not None,
        ROW_REPEAT=row_repeat,
        SPLIT=_TOPK_SPLIT,
        num_warps=_TOPK_WARPS,
    )
    if _TOPK_SPLIT:
        _candidate_sort_kernel[(rows,)](
            scratch,
            meta,
            out,
            *out.stride(),
            k,
            OUT_K=out_k,
            KPAD=kpad,
            num_warps=_TOPK_SORT_WARPS,
        )


@triton.jit(do_not_specialize=["width", "nblocks"])
def _candidate_flags_kernel(
    candidates,
    starts,
    flags,
    stride_row,
    stride_col,
    stride_start,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    K: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, 1024)
    for tile in range(tl.cdiv(nblocks + 1, 1024)):
        slots = tile * 1024 + offsets
        tl.store(flags + row * (nblocks + 1) + slots, 0, slots <= nblocks)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    cols = tl.arange(0, triton.next_power_of_2(K))
    block = tl.load(
        candidates + row * stride_row + cols * stride_col, cols < K, other=-1
    ).to(tl.int64)
    # Preserve the packed-column clamp for candidates beyond the logits width.
    block = tl.where(start + block * BLOCK_SIZE >= width, nblocks, block)
    tl.debug_barrier()
    tl.store(flags + row * (nblocks + 1) + block, 1, (cols < K) & (block >= 0))


@triton.jit(do_not_specialize=["width", "nblocks"])
def _mask_candidates_kernel(
    logits,
    starts,
    ends,
    flags,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    valid = (cols >= start) & (cols < end) & (cols < width)
    block = (cols - start) // BLOCK_SIZE
    keep = tl.load(flags + row * (nblocks + 1) + block, valid, other=0)
    edge = tl.load(flags + row * (nblocks + 1) + nblocks)
    keep = (keep != 0) | ((cols == width - 1) & (edge != 0))
    tl.store(
        logits + row * stride_row + cols * stride_col,
        -float("inf"),
        (cols < width) & ~(valid & keep),
    )


@triton.jit(do_not_specialize=["width", "nblocks"])
def _mask_candidates_bounded_kernel(
    logits,
    starts,
    ends,
    flags,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
    NPROG: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    pid = tl.program_id(1)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    stop = tl.minimum(end, width)
    edge = tl.load(flags + row * (nblocks + 1) + nblocks)
    ntiles = tl.cdiv(tl.maximum(stop - start, 0), TILE)
    for t in range(pid, ntiles, NPROG):
        cols = start + t * TILE + tl.arange(0, TILE)
        valid = cols < stop
        block = (cols - start) // BLOCK_SIZE
        keep = tl.load(flags + row * (nblocks + 1) + block, valid, other=0)
        keep = (keep != 0) | ((cols == width - 1) & (edge != 0))
        tl.store(
            logits + row * stride_row + cols * stride_col,
            -float("inf"),
            valid & ~keep,
        )


def select_candidate_blocks(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    topk_blocks: int,
    block_size: int,
    out: torch.Tensor,
    row_repeat: int = 1,
    max_row_len: int | None = None,
) -> None:
    """Select local block IDs by maximum score, pinning each row's newest block.

    Row bounds are in packed column space; absent starts mean zero.
    Decode rows share bounds in groups of ``row_repeat``. Output is -1 padded.

    ``max_row_len`` is an optional host-side upper bound on every row's
    ``end - start``. Logits are allocated ``max_model_len`` wide, so without it
    the scores and the top-k sweep that whole width. Blocks past a row's end
    score -inf and are never kept, and top-k breaks ties by lowest index, so
    the bound leaves the selected set unchanged.
    """
    assert logits.is_cuda
    rows, width = logits.shape
    if not rows:
        return
    if not width:
        out.fill_(-1)
        return
    score_width = width if max_row_len is None else max(1, min(width, max_row_len))
    nblocks = triton.cdiv(score_width, block_size)
    scores = logits.new_empty((rows, nblocks))
    # torch.topk(sorted=True) sorts k <= 128 results with a comparison sort
    # (bitonic below 32, unstable; warp merge sort up to 128, NaN largest) and
    # 128 < k <= 4096 with a stable CUB radix sort; the fast kernel reproduces
    # the radix-sort regime only (DSV4.1: k = 2048).
    fast = (
        FAST_CANDIDATE_TOPK
        and topk_blocks == out.shape[1]
        and 128 < min(topk_blocks, nblocks)
        and triton.next_power_of_2(topk_blocks) <= 2048
        and nblocks < (1 << 21) - 1
    )
    _block_scores_kernel[(rows, triton.cdiv(nblocks, 128))](
        logits,
        row_ks,
        row_ke,
        scores,
        *logits.stride(),
        row_ks.stride(0) if row_ks is not None else 0,
        row_ke.stride(0),
        width,
        nblocks,
        block_size,
        row_ks is not None,
        row_repeat,
        128,
        SKIP_TAIL=fast,
    )
    if fast:
        fast_select_topk(
            scores,
            min(topk_blocks, nblocks),
            out,
            row_ks,
            row_ke,
            width,
            block_size,
            row_repeat,
        )
        return
    # Keep the existing top-k tie behavior.
    top = scores.topk(min(topk_blocks, nblocks), dim=-1)
    _store_candidates_kernel[(rows, triton.cdiv(topk_blocks, 256))](
        top.values,
        top.indices,
        out,
        *out.stride(),
        top.values.shape[1],
        topk_blocks,
        256,
    )


def apply_candidate_mask(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    candidate_blocks: torch.Tensor,
    block_size: int,
    row_repeat: int = 1,
) -> None:
    """Mask packed logits outside causal bounds and request-local candidates."""
    assert logits.is_cuda
    rows, width = logits.shape
    if not rows or not width:
        return
    nblocks = triton.cdiv(width, block_size)
    flags = torch.empty((rows, nblocks + 1), device=logits.device, dtype=torch.uint8)
    start_stride = row_ks.stride(0) if row_ks is not None else 0
    _candidate_flags_kernel[(rows,)](
        candidate_blocks,
        row_ks,
        flags,
        *candidate_blocks.stride(),
        start_stride,
        width,
        nblocks,
        block_size,
        candidate_blocks.shape[1],
        row_ks is not None,
        row_repeat,
    )
    if BOUNDED_CANDIDATE_MASK:
        nprog = max(1, min(_BOUNDED_MASK_PROGS, triton.cdiv(width, 1024)))
        _mask_candidates_bounded_kernel[(rows, nprog)](
            logits,
            row_ks,
            row_ke,
            flags,
            *logits.stride(),
            start_stride,
            row_ke.stride(0),
            width,
            nblocks,
            block_size,
            row_ks is not None,
            row_repeat,
            1024,
            nprog,
        )
        return
    _mask_candidates_kernel[(rows, triton.cdiv(width, 1024))](
        logits,
        row_ks,
        row_ke,
        flags,
        *logits.stride(),
        start_stride,
        row_ke.stride(0),
        width,
        nblocks,
        block_size,
        row_ks is not None,
        row_repeat,
        1024,
    )
