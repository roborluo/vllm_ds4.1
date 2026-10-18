# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton

# Columns one mask program covers per loop iteration.
_MASK_TILE = 1024
# Mask programs to spread over the rows of one launch: enough to keep the GPU
# busy on fully live rows, without a program per tile of every short row. While
# the tiles of all rows fit, every tile gets its own program, as in a full-width
# sweep; past that, each row gets ceil(budget / rows) programs that stride over
# its tiles.
_MASK_PROGRAM_BUDGET = 16384


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
):
    row = tl.program_id(0).to(tl.int64)
    blocks = tl.program_id(1) * TILE + tl.arange(0, TILE)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
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
    ends=None,
    stride_end=0,
    HAS_ENDS: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    row_flags = flags + row * (nblocks + 1)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    # Load the candidates first so that the load overlaps the clear below.
    cols = tl.arange(0, triton.next_power_of_2(K))
    block = tl.load(
        candidates + row * stride_row + cols * stride_col, cols < K, other=-1
    )
    # Slot nblocks is the edge flag. Given the row ends, only the slots the
    # mask reads are cleared: the blocks covering [start, min(end, width)) and
    # the edge slot. Callers that pass no ends get every slot cleared.
    if HAS_ENDS:
        end = tl.load(ends + row // ROW_REPEAT * stride_end)
        live = tl.cdiv(tl.maximum(tl.minimum(end, width) - start, 0), BLOCK_SIZE)
    else:
        live = nblocks
    offsets = tl.arange(0, 1024)
    for tile in range(tl.cdiv(live, 1024)):
        slots = tile * 1024 + offsets
        tl.store(row_flags + slots, 0, slots < live)
    tl.store(row_flags + nblocks, 0)
    # Preserve the packed-column clamp for candidates beyond the logits width.
    block = block.to(tl.int64)
    block = tl.where(start + block * BLOCK_SIZE >= width, nblocks, block)
    tl.debug_barrier()
    # Candidates past the cleared blocks set slots that the mask never reads.
    tl.store(row_flags + block, 1, (cols < K) & (block >= 0))


@triton.jit
def _mask_candidates_tile(
    row_logits,
    row_flags,
    cols,
    start,
    stop,
    edge,
    stride_col,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    written = cols < stop
    valid = written & (cols >= start)
    # Lanes that load have cols >= start, where the unsigned quotient is exact
    # and cheaper than the signed one.
    block = (cols - start).to(tl.uint32) // BLOCK_SIZE
    keep = tl.load(row_flags + block, valid, other=0)
    keep = (keep != 0) | ((cols == width - 1) & (edge != 0))
    tl.store(
        row_logits + cols * stride_col,
        -float("inf"),
        written & ~(valid & keep),
    )


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
    row_logits = logits + row * stride_row
    row_flags = flags + row * (nblocks + 1)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    edge = tl.load(row_flags + nblocks)
    # Only columns below stop are written.
    stop = tl.minimum(end, width)
    if tl.num_programs(1) * TILE >= width:
        # One program per tile, as in a full-width sweep.
        cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
        _mask_candidates_tile(
            row_logits,
            row_flags,
            cols,
            start,
            stop,
            edge,
            stride_col,
            width,
            BLOCK_SIZE,
        )
    else:
        # The programs of a row stride over its tiles, so the grid does not
        # depend on the row ends.
        step = tl.num_programs(1) * TILE
        for tile_start in range(tl.program_id(1) * TILE, stop, step):
            cols = tile_start + tl.arange(0, TILE)
            _mask_candidates_tile(
                row_logits,
                row_flags,
                cols,
                start,
                stop,
                edge,
                stride_col,
                width,
                BLOCK_SIZE,
            )


def select_candidate_blocks(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    topk_blocks: int,
    block_size: int,
    out: torch.Tensor,
    row_repeat: int = 1,
) -> None:
    """Select local block IDs by maximum score, pinning each row's newest block.

    Row bounds are in packed column space; absent starts mean zero.
    Decode rows share bounds in groups of ``row_repeat``. Output is -1 padded.
    """
    assert logits.is_cuda
    rows, width = logits.shape
    if not rows:
        return
    if not width:
        out.fill_(-1)
        return
    nblocks = triton.cdiv(width, block_size)
    scores = logits.new_empty((rows, nblocks))
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
    )
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
    """Mask packed logits outside causal bounds and request-local candidates.

    Only each row's ``[0, min(end, width))`` is rewritten: -inf before
    ``start`` and on blocks that are not candidates, the logit unchanged on
    candidate blocks. Later columns are left as they are; every top-k that
    follows reads a row only within that range. Decode logits rows are sized
    for the longest context, so the cost follows each row's end, not the row
    width. Row bounds are in packed column space; absent starts mean zero.
    Decode rows share bounds in groups of ``row_repeat``.
    """
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
        ends=row_ke,
        stride_end=row_ke.stride(0),
        HAS_ENDS=True,
    )
    # The grid depends only on the logits shape, so a captured CUDA graph
    # stays valid as the row ends change.
    splits = min(
        triton.cdiv(width, _MASK_TILE), triton.cdiv(_MASK_PROGRAM_BUDGET, rows)
    )
    _mask_candidates_kernel[(rows, splits)](
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
        _MASK_TILE,
    )
