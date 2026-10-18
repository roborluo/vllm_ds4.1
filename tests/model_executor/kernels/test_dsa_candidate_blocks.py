# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the bounded DSA candidate mask.

``apply_candidate_mask`` rewrites each row's ``[0, min(end, width))`` with the
values a full-width sweep writes there and leaves the later columns untouched.
The programs of a row stride over its tiles, so the launch grid depends only on
the logits shape and a captured CUDA graph must follow new row ends.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Needs a CUDA or ROCm GPU."
)

DEVICE = "cuda"
NUM_CANDIDATES = 64


def _reference_flags(candidates, starts, width, block_size):
    """Flag slots after a full clear and the candidate scatter.

    Slot ``b`` marks candidate block ``b``. Slot ``nblocks`` marks candidates
    whose block starts at or past the width; they keep only the last column.
    """
    rows = candidates.shape[0]
    nblocks = (width + block_size - 1) // block_size
    blocks = candidates.long()
    slots = torch.where(starts[:, None] + blocks * block_size >= width, nblocks, blocks)
    flags = torch.zeros(rows, nblocks + 1, dtype=torch.uint8, device=DEVICE)
    row_ids = torch.arange(rows, device=DEVICE)[:, None].expand_as(slots)
    picked = slots >= 0
    flags[row_ids[picked], slots[picked]] = 1
    return flags


def _reference_mask(logits, starts, ends, candidates, block_size):
    """Expected logits for per-row int64 ``starts`` and ``ends``."""
    width = logits.shape[1]
    nblocks = (width + block_size - 1) // block_size
    flags = _reference_flags(candidates, starts, width, block_size).bool()
    cols = torch.arange(width, device=DEVICE)
    written = cols < ends[:, None]
    valid = written & (cols >= starts[:, None])
    blocks = (cols - starts[:, None]).clamp(min=0) // block_size
    keep = flags.gather(1, blocks) | ((cols == width - 1) & flags[:, nblocks:])
    masked = torch.where(valid & keep, logits, -torch.inf)
    return torch.where(written, masked, logits)


def _assert_bitwise_equal(actual, expected):
    differ = actual.view(torch.int32) != expected.view(torch.int32)
    assert not differ.any(), (
        f"{int(differ.sum())} values differ, first at {differ.nonzero()[0].tolist()}"
    )


def _row_bounds(rows, width, block_size, row_repeat, with_starts):
    """Per-group starts and ends: full, empty, past-the-width and ragged rows.

    The first row is always fully live.
    """
    groups = rows // row_repeat
    pattern = [width, 0, width + 5, 1, width - 1, block_size + 1, 1025]
    dtype = torch.int64 if with_starts else torch.int32
    ends = torch.tensor(
        [pattern[g % len(pattern)] for g in range(groups)], dtype=dtype, device=DEVICE
    )
    starts = None
    if with_starts:
        starts = torch.randint(0, width // 2, (groups,), dtype=dtype, device=DEVICE)
    return starts, ends


def _per_row(bounds, row_repeat, groups):
    if bounds is None:
        return torch.zeros(groups * row_repeat, dtype=torch.int64, device=DEVICE)
    return bounds.long().repeat_interleave(row_repeat)


@pytest.mark.parametrize(
    "rows,width,block_size,row_repeat,with_starts,col_stride",
    [
        # One fully live row as wide as a long-context decode row.
        (1, 1 << 20, 8, 1, False, 1),
        # Prefill-style starts on a ragged, column-strided logits view.
        (6, 4099, 3, 1, True, 2),
        # Speculative decode: groups of row_repeat rows share their bounds.
        (8, 70001, 8, 2, False, 1),
        # More tiles than the program budget: programs stride over tiles.
        (20, 1 << 20, 8, 1, True, 1),
    ],
)
def test_apply_candidate_mask_matches_reference(
    rows, width, block_size, row_repeat, with_starts, col_stride
):
    """Bitwise equal on [0, min(end, width)), untouched past it, eager and
    under CUDA graph replay with row ends other than the captured ones."""
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        apply_candidate_mask,
    )

    torch.manual_seed(0)
    groups = rows // row_repeat
    nblocks = (width + block_size - 1) // block_size
    logits = torch.randn(rows, width * col_stride, device=DEVICE)[:, ::col_stride]
    logits[0, 5] = float("nan")
    starts, ends = _row_bounds(rows, width, block_size, row_repeat, with_starts)
    candidates = torch.randint(
        -1, nblocks, (rows, NUM_CANDIDATES), dtype=torch.int32, device=DEVICE
    )
    candidates[:, 0] = nblocks + 10
    candidates[:, 1] = -1
    candidates[:, 2] = 0
    row_starts = _per_row(starts, row_repeat, groups)

    expected = _reference_mask(
        logits, row_starts, _per_row(ends, row_repeat, groups), candidates, block_size
    )
    apply_candidate_mask(logits, starts, ends, candidates, block_size, row_repeat)
    _assert_bitwise_equal(logits, expected)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        apply_candidate_mask(logits, starts, ends, candidates, block_size, row_repeat)
    # Fully live rows become partial and the others fully live.
    ends.copy_(torch.where(ends >= width, width // 3, width))
    logits.copy_(torch.randn(rows, width, device=DEVICE))
    expected = _reference_mask(
        logits, row_starts, _per_row(ends, row_repeat, groups), candidates, block_size
    )
    graph.replay()
    _assert_bitwise_equal(logits, expected)


@pytest.mark.parametrize("with_ends", [False, True])
@pytest.mark.parametrize("with_starts", [False, True])
def test_candidate_flags_clear_the_slots_the_mask_reads(with_starts, with_ends):
    """Without ends, as ``_apply_candidate_mask_strided`` launches it with
    positional arguments only, every slot is cleared. With ends, the slots of
    the blocks covering [start, min(end, width)) and the edge slot are."""
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        _candidate_flags_kernel,
    )

    torch.manual_seed(0)
    rows, width, block_size = 5, 9001, 3
    nblocks = (width + block_size - 1) // block_size
    starts = None
    if with_starts:
        starts = torch.tensor(
            [0, 7, width - 1, 100, 0], dtype=torch.int32, device=DEVICE
        )
    ends = torch.tensor(
        [width, 1, width + 5, 2000, 0], dtype=torch.int32, device=DEVICE
    )
    candidates = torch.randint(
        -1, nblocks, (rows, NUM_CANDIDATES), dtype=torch.int32, device=DEVICE
    )
    candidates[:, 0] = nblocks + 10
    # Stale bytes, as torch.empty may return them.
    flags = torch.full((rows, nblocks + 1), 0xAB, dtype=torch.uint8, device=DEVICE)
    args = (
        candidates,
        starts,
        flags,
        *candidates.stride(),
        starts.stride(0) if starts is not None else 0,
        width,
        nblocks,
        block_size,
        NUM_CANDIDATES,
        starts is not None,
        1,
    )
    row_starts = _per_row(starts, 1, rows)
    if with_ends:
        _candidate_flags_kernel[(rows,)](
            *args, ends=ends, stride_end=ends.stride(0), HAS_ENDS=True
        )
        live = (ends.long().clamp(max=width) - row_starts).clamp(min=0)
        live = (live + block_size - 1) // block_size
        slots = torch.arange(nblocks + 1, device=DEVICE)
        read = (slots < live[:, None]) | (slots == nblocks)
    else:
        _candidate_flags_kernel[(rows,)](*args)
        read = torch.ones_like(flags, dtype=torch.bool)
    expected = _reference_flags(candidates, row_starts, width, block_size)
    assert torch.equal(flags[read], expected[read])
