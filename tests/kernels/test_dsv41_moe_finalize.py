# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1's fused MoE top-k finalize + shared-expert add.

The fused kernel replaces the TRT-LLM finalize and the shared-expert add that
follows it, so it must reproduce them bit for bit. ``moe_unpermute`` is that
finalize: ``check_deferred_moe_finalize`` holds it to the TRTLLM-Gen experts'
own finalized output.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.moe_output import (
    MoEOutput,
    UnfinalizedMoEOutput,
)
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    moe_permute_unpermute_supported,
    moe_unpermute,
)
from vllm.models.deepseek_v41.nvidia.model import _finalize_shared_add
from vllm.platforms import current_platform


@pytest.mark.parametrize("num_tokens", [1, 17, 1000])
# DeepSeek V4.1's shape, and a hidden size that ends in a partial block.
@pytest.mark.parametrize("hidden_size,top_k", [(5120, 6), (2880, 4)])
def test_finalize_shared_add_matches_finalize_then_add(
    num_tokens: int, hidden_size: int, top_k: int
):
    if not current_platform.is_cuda() or not moe_permute_unpermute_supported():
        pytest.skip("moe_unpermute requires CUDA")
    torch.manual_seed(0)
    num_routes = num_tokens * top_k
    # Rows past the routed ones stand in for the kernel's padding.
    gemm2_permuted = torch.randn(
        num_routes + 8, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    permuted_idx = torch.randperm(num_routes, device="cuda").to(torch.int32)
    permuted_idx = permuted_idx.view(num_tokens, top_k)
    expert_weights = torch.rand(num_tokens, top_k, device="cuda").softmax(dim=-1)
    shared_output = torch.randn(
        num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    # One route goes to another rank's expert (-1), which the finalize skips.
    # moe_unpermute's input keeps a row for that route with zero weight
    # instead, which leaves its sum unchanged.
    unfinalized_idx = permuted_idx.clone()
    unfinalized_idx[0, -1] = -1
    reference_weights = expert_weights.clone()
    reference_weights[0, -1] = 0

    routed_output = torch.empty_like(shared_output)
    moe_unpermute(
        routed_output,
        gemm2_permuted,
        reference_weights,
        permuted_idx,
        # The kernel reads its valid-row count through this pointer either way.
        expert_first_token_offset=gemm2_permuted.new_full(
            (1,), gemm2_permuted.shape[0], dtype=torch.int64
        ),
    )
    expected = shared_output + routed_output

    actual = _finalize_shared_add(
        MoEOutput(
            routed=UnfinalizedMoEOutput(
                gemm2_permuted, expert_weights, unfinalized_idx
            ),
            shared_output=shared_output,
        )
    )
    torch.testing.assert_close(
        actual.view(torch.int16), expected.view(torch.int16), atol=0, rtol=0
    )
