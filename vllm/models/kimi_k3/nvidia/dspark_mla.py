# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K3 dense MLA draft model for DSpark speculative decoding."""

from collections.abc import Iterable

import torch
import torch.nn as nn

import vllm._custom_ops as ops
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.qwen3_dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    get_draft_quant_config,
    maybe_prefix,
)
from vllm.models.common.ops.fused_allreduce_rms_norm import fused_allreduce_rms_norm
from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention
from vllm.models.kimi_k3.nvidia.model import KimiMLP
from vllm.v1.kv_cache_interface import is_quantized_kv_cache
from vllm.v1.worker.workspace import current_workspace_manager

_GROUPED_KV_CACHE_DTYPES = frozenset(
    {"auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2"}
)


def _duplicate_context_kv_weights(
    weights: Iterable[tuple[str, torch.Tensor]], num_layers: int
) -> Iterable[tuple[str, torch.Tensor]]:
    """Load each layer's KV projection into the cross-layer linear."""
    for name, weight in weights:
        yield name, weight
        layer_prefix, marker, param_name = name.partition(
            ".self_attn.kv_a_proj_with_mqa."
        )
        if not marker:
            continue
        layer_idx_str = layer_prefix.rsplit(".", 1)[-1]
        if not layer_idx_str.isdecimal():
            continue
        layer_idx = int(layer_idx_str)
        if layer_idx >= num_layers:
            continue
        fused_weight = weight.detach()
        fused_weight.shard_id = layer_idx
        yield f"context_kv_proj.{param_name}", fused_weight


class K3DSparkDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        layer_idx: int,
        start_layer_id: int,
        prefix: str,
    ) -> None:
        super().__init__()
        quant_config = get_draft_quant_config(vllm_config)
        self.self_attn = MultiHeadLatentAttention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(
                prefix, f"layers.{start_layer_id + layer_idx}.self_attn"
            ),
            use_rope=True,
            non_causal_multi_token_decode=True,
        )
        # Both row-parallel outputs stay un-reduced; their all-reduces are fused
        # into the RMSNorm that follows via fused_allreduce_rms_norm.
        self.self_attn.o_proj.reduce_results = False
        self.mlp = KimiMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            reduce_results=False,
            prefix=maybe_prefix(prefix, f"layers.{start_layer_id + layer_idx}.mlp"),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            # First layer: hidden_states is the (already reduced) embedding.
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = fused_allreduce_rms_norm(
                hidden_states, residual, self.input_layernorm
            )

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states, residual = fused_allreduce_rms_norm(
            hidden_states, residual, self.post_attention_layernorm
        )
        # The MLP output is reduced by the next layer's input_layernorm (or by
        # the model's final_norm).
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class K3DSparkModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int,
        prefix: str,
    ) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.quant_config = get_draft_quant_config(vllm_config)

        # The frozen target embedding is aliased after the draft checkpoint loads.
        self.embed_tokens: nn.Module | None = None

        self.context_proj = ReplicatedLinear(
            self.config.target_hidden_size * self.config.num_target_layers,
            self.config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "context_proj"),
        )
        self.context_norm = RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )

        self.layers = nn.ModuleList(
            [
                K3DSparkDecoderLayer(
                    vllm_config=vllm_config,
                    config=self.config,
                    layer_idx=layer_idx,
                    start_layer_id=start_layer_id,
                    prefix=prefix,
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        kv_width = self.config.kv_lora_rank + self.config.qk_rope_head_dim
        self.context_kv_proj = MergedColumnParallelLinear(
            self.config.hidden_size,
            [kv_width] * self.config.num_hidden_layers,
            bias=False,
            return_bias=False,
            quant_config=self.quant_config,
            prefix=maybe_prefix(
                prefix,
                f"layers.{start_layer_id}.self_attn.fused_qkv_a_proj",
            ),
            disable_tp=True,
        )
        self.final_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.markov_head = DSparkMarkovHead(
            self.config.vocab_size,
            self.config.draft_vocab_size,
            self.config.markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )
        self.confidence_head: DSparkConfidenceHead | None = None
        if getattr(self.config, "enable_confidence_head", False):
            with_markov = getattr(self.config, "confidence_head_with_markov", False)
            input_dim = self.config.hidden_size + (
                self.config.markov_rank if with_markov else 0
            )
            self.confidence_head = DSparkConfidenceHead(
                input_dim,
                prefix=maybe_prefix(prefix, "confidence_head"),
                bias=True,
                with_markov=with_markov,
            )
        self._max_num_context_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.embed_tokens is not None
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.context_norm(self.context_proj(hidden_states))

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        """Project target-derived context into each draft layer's latent cache."""
        if not hasattr(self, "_num_context_layers"):
            self._build_fused_context_kv_metadata()
        self._precompute_fused_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def _build_fused_context_kv_metadata(self) -> None:
        """Build cross-layer metadata after checkpoint loading."""
        attentions = [layer.self_attn for layer in self.layers]
        assert attentions
        attn0 = attentions[0]
        assert attn0.q_lora_rank is not None
        kv_width = attn0.kv_lora_rank + attn0.qk_rope_head_dim
        for attn in attentions:
            assert attn.q_lora_rank is not None
            assert (
                attn.q_lora_rank == attn0.q_lora_rank
                and attn.kv_lora_rank == attn0.kv_lora_rank
                and attn.qk_rope_head_dim == attn0.qk_rope_head_dim
                and attn.kv_a_layernorm.variance_epsilon
                == attn0.kv_a_layernorm.variance_epsilon
            ), "All MLA DSpark layers must share their latent KV geometry."
        self._context_kv_norm_weights = torch.stack(
            [attn.kv_a_layernorm.weight.detach() for attn in attentions], dim=0
        ).contiguous()
        self._num_context_layers = len(attentions)
        self._context_kv_width = kv_width
        self._context_kv_lora_rank = attn0.kv_lora_rank
        self._context_rope_dim = attn0.qk_rope_head_dim
        self._context_rms_norm_eps = attn0.kv_a_layernorm.variance_epsilon
        self._context_kv_scales: torch.Tensor | None = None
        if attn0.kv_cache_dtype in _GROUPED_KV_CACHE_DTYPES and is_quantized_kv_cache(
            attn0.kv_cache_dtype
        ):
            self._context_kv_scales = torch.stack(
                [attn._k_scale.reshape(()) for attn in attentions]
            )

    def _precompute_fused_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None,
    ) -> None:
        num_ctx = context_states.shape[0]
        num_layers = self._num_context_layers

        # One KV-only GEMM replaces five full Q+KV GEMMs. For K3 this projects
        # 5*576 rows rather than 5*2112 rows (72.7% fewer A-projection FLOPs).
        all_kv = self.context_kv_proj(context_states)
        all_kv = all_kv.view(num_ctx, num_layers, self._context_kv_width)
        all_kv_c = all_kv[..., : self._context_kv_lora_rank]
        all_k_pe = all_kv[..., self._context_kv_lora_rank :]

        # Layer-major layout lets the 2-D RMSNorm weights select a distinct row
        # for each draft layer in one grouped kernel.
        all_kv_c = all_kv_c.permute(1, 0, 2).contiguous()
        all_kv_c_normed = torch.empty_like(all_kv_c)
        ops.rms_norm(
            all_kv_c_normed,
            all_kv_c,
            self._context_kv_norm_weights,
            self._context_rms_norm_eps,
        )

        all_k_pe = all_k_pe.permute(1, 0, 2).contiguous()
        all_k_pe_flat = all_k_pe.view(num_layers * num_ctx, 1, self._context_rope_dim)
        (repeated_positions,) = current_workspace_manager().get_simultaneous(
            ((num_layers * self._max_num_context_tokens,), torch.int64),
        )
        repeated_positions = repeated_positions[: num_layers * num_ctx]
        repeated_positions.view(num_layers, num_ctx).copy_(context_positions)
        # Keep the single-tensor context RoPE on vLLM's optimized CUDA op;
        # DeepSeek YaRN's FlashInfer wrapper assumes a non-null key tensor.
        rotary_emb = self.layers[0].self_attn.rotary_emb
        assert rotary_emb is not None
        ops.rotary_embedding(
            repeated_positions,
            all_k_pe_flat,
            None,
            rotary_emb.head_size,
            rotary_emb.cos_sin_cache,
            rotary_emb.is_neox_style,
        )
        all_k_pe = all_k_pe_flat.view(num_layers, num_ctx, 1, self._context_rope_dim)

        if context_slot_mapping is None:
            return

        cache_layers = [layer.self_attn for layer in self.layers]
        cache_dtype = cache_layers[0].kv_cache_dtype
        if (
            cache_dtype in _GROUPED_KV_CACHE_DTYPES
            and all(cl.kv_cache_dtype == cache_dtype for cl in cache_layers)
            and self._has_uniform_block_layout(cache_layers)
            and (
                isinstance(context_slot_mapping, torch.Tensor)
                or all(s is not None for s in context_slot_mapping)
            )
        ):
            # Grouped context KV insert assumes that all layers share the same
            # cache dtype and block layout.

            if isinstance(context_slot_mapping, (list, tuple)):
                per_layer_slot_mappings = [
                    s for s in context_slot_mapping if s is not None
                ]
                if len({s.data_ptr() for s in per_layer_slot_mappings}) == 1:
                    # All rows alias to the same slot mapping.
                    slot_mapping = (
                        per_layer_slot_mappings[0].unsqueeze(0).expand(num_layers, -1)
                    )
                else:
                    slot_mapping = torch.stack(per_layer_slot_mappings, dim=0)
            else:
                # Broadcast the single shared context_slot_mapping tensor.
                slot_mapping = context_slot_mapping.unsqueeze(0).expand(num_layers, -1)

            ref_cache = cache_layers[0].kv_cache
            ops.concat_and_cache_mla_grouped(
                all_kv_c_normed,
                all_k_pe.squeeze(2),
                self._get_context_kv_cache_ptrs(cache_layers),
                slot_mapping,
                ref_cache.size(1),
                ref_cache.stride(0),
                ref_cache.stride(1),
                self._context_kv_scales,
                cache_dtype,
            )
            return

        for layer_idx, layer in enumerate(self.layers):
            slot_mapping = (
                context_slot_mapping[layer_idx]
                if isinstance(context_slot_mapping, (list, tuple))
                else context_slot_mapping
            )
            if slot_mapping is None:
                continue
            attn = layer.self_attn
            attn.impl.do_kv_cache_update(
                all_kv_c_normed[layer_idx],
                all_k_pe[layer_idx],
                attn.kv_cache,
                slot_mapping,
                attn.kv_cache_dtype,
                attn._k_scale,
            )

    def _has_uniform_block_layout(
        self,
        cache_layers: list[MultiHeadLatentAttention],
    ) -> bool:
        key = tuple(cl.kv_cache.data_ptr() for cl in cache_layers)
        if getattr(self, "_kv_cache_ptrs_key", None) != key:
            assert not torch.cuda.is_current_stream_capturing()
            self._kv_cache_ptrs_key = key
            if hasattr(self, "_context_cache_ptrs"):
                del self._context_cache_ptrs
            ref_cache = cache_layers[0].kv_cache
            self._layers_share_kv_block_layout = all(
                cl.kv_cache.size(1) == ref_cache.size(1)
                and cl.kv_cache.stride(0) == ref_cache.stride(0)
                and cl.kv_cache.stride(1) == ref_cache.stride(1)
                for cl in cache_layers
            )
        return self._layers_share_kv_block_layout

    def _get_context_kv_cache_ptrs(
        self,
        cache_layers: list[MultiHeadLatentAttention],
    ) -> torch.Tensor:
        # The per-layer KV cache base pointers are stable after allocation, so
        # build the pointer array once and return it on every call.
        if not hasattr(self, "_context_cache_ptrs"):
            ref_cache = cache_layers[0].kv_cache
            cache_ptrs = torch.tensor(
                [cl.kv_cache.data_ptr() for cl in cache_layers],
                dtype=torch.int64,
                device=ref_cache.device,
            )
            self._context_cache_ptrs = cache_ptrs
        return self._context_cache_ptrs

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)

        hidden_states = inputs_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        hidden_states, _ = fused_allreduce_rms_norm(
            hidden_states, residual, self.final_norm
        )
        return hidden_states


class K3DSparkForCausalLM(nn.Module):
    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None
    hf_to_vllm_mapper = WeightsMapper(
        # confidence_head is training-only. The frozen target embedding and LM
        # head are shared after this draft-specific checkpoint is loaded.
        orig_to_new_substr={
            "confidence_head": None,
            "embed_tokens": None,
            "lm_head": None,
        },
        orig_to_new_prefix={"": "model."},
        orig_to_new_stacked={
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
            ".q_a_proj": (".fused_qkv_a_proj", 0),
            ".kv_a_proj_with_mqa": (".fused_qkv_a_proj", 1),
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "enable_confidence_head", False):
            # The class mapper drops the head as training-only; keep it for drafts
            # that actually built one, since adaptive verification reads it.
            self.hf_to_vllm_mapper = self.hf_to_vllm_mapper | WeightsMapper(
                orig_to_new_substr={"confidence_head": "confidence_head"}
            )
        target_layer_num = vllm_config.model_config.get_total_num_hidden_layers()
        self.model = K3DSparkModel(
            vllm_config=vllm_config,
            start_layer_id=target_layer_num,
            prefix=maybe_prefix(prefix, "model"),
        )

        # Assigned by load_dspark_model from the target. Keeping no placeholder
        # avoids a transient full-vocabulary allocation for this 163k-vocab model.
        self.lm_head: nn.Module | None = None
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(hidden_states)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.layer_name for layer in self.model.layers]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.lm_head is not None
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.compute_logits(hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def compute_confidence(
        self, head_hidden: torch.Tensor, markov_embed: torch.Tensor
    ) -> torch.Tensor:
        """Per-position acceptance probability for each drafted token."""
        assert self.model.confidence_head is not None
        return torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        # read: 1. all weights. 2. context kv weights
        weights = _duplicate_context_kv_weights(weights, len(self.model.layers))
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def process_weights_after_loading(self) -> None:
        self.model._build_fused_context_kv_metadata()


# ---------------------------------------------------------------------------
# [DFlash2-MLA overlay] MLA DFlash2 drafter for Kimi-K3
# (RadixArk/Kimi-K3.1-Dflash2-Preview). Everything below is new and is reached
# only through the "DFlash2K3MLADraftModel" registry entry, which
# SpeculativeConfig.hf_config_override selects for a DFlash2DraftModel whose
# dflash_config.attention_mode is "mla". Served under method="dflash", so the
# DFlash2Speculator does the drafting: K+1 query slots (slot 0 = bonus/current
# token, drafts sampled from slots 1..K), block-causal attention and the
# candidate-selector walk. The DSpark classes above are untouched.
#
# Relative to K3DSparkModel this adds the DFlash2 block convolutions
# (attention_conv / mlp_conv) and the candidate selector, sizes the context
# projection from dflash_config.target_layer_ids and drops the DSpark markov /
# confidence heads. Context latents (precompute_and_store_context_kv) are not
# convolved, as in the GQA DFlash2 drafter.
# ---------------------------------------------------------------------------
import os
import re

from vllm.logger import init_logger

_df2_logger = init_logger(__name__)


def _dflash2_mla_conv_enabled() -> bool:
    # A/B switch: VLLM_DFLASH2_MLA_CONV=0 skips the block convolutions (the
    # weights still load). Default on.
    return os.getenv("VLLM_DFLASH2_MLA_CONV", "1") != "0"


def _dflash2_mla_window_enabled() -> bool:
    # A/B switch: VLLM_DFLASH2_MLA_WINDOW=0 drops the checkpoint's sliding
    # window (the draft then reads the whole context, the pre-window overlay
    # behaviour). Default on.
    return os.getenv("VLLM_DFLASH2_MLA_WINDOW", "1") != "0"


# Decode backends whose forward_mqa reads only decode.block_table,
# decode.seq_lens, decode.dcp_tot_seq_lens and the max_seq_len upper bound, so a
# windowed copy of those tensors is all they need. Backends that plan from
# seq_lens in the metadata builder (e.g. FlashMLA tile scheduling) are excluded.
_DFLASH2_MLA_WINDOW_BACKENDS = frozenset({"TOKENSPEED_MLA", "TRITON_MLA"})
_df2_window_warned: set[str] = set()


def _df2_window_warn_once(key: str, msg: str, *args, error: bool = True) -> None:
    if key not in _df2_window_warned:
        _df2_window_warned.add(key)
        (_df2_logger.error if error else _df2_logger.warning)(msg, *args)


def _window_mla_decode_metadata(
    attn_metadata, window_left: int, block_size: int, cp_size: int
):
    """Copy of MLA decode metadata with each request's KV start moved forward to
    the last ``window_left`` context positions before its first query token.

    The vLLM equivalent of TRT's ``_window_mla_block_fixup``: whole block-table
    entries are dropped from the front (one entry covers ``block_size * cp_size``
    global positions under DCP), and seq_lens / dcp_tot_seq_lens drop by the same
    amount. The start is rounded down to an entry boundary, so every query sees
    a superset of its exact window (at most ``block_size * cp_size - 1`` extra
    older keys). Cached k_pe is already rotated, so positions need no change,
    and intra-block causality is unchanged because only a common prefix is
    dropped. Under DCP the dropped prefix is a multiple of
    ``cp_size * interleave`` (block_size is a multiple of the interleave), so
    every rank drops exactly ``skip * block_size`` local tokens.

    Returns None (caller runs unwindowed) when there is nothing to window.
    Pure tensor ops, no host sync: safe inside a FULL CUDA graph capture.
    """
    import copy

    decode = getattr(attn_metadata, "decode", None)
    num_decodes = int(getattr(attn_metadata, "num_decodes", 0) or 0)
    if decode is None or num_decodes <= 0:
        return None
    num_decode_tokens = int(attn_metadata.num_decode_tokens)
    if int(getattr(attn_metadata, "num_actual_tokens", 0) or 0) > num_decode_tokens:
        _df2_window_warn_once(
            "prefill",
            "DFlash2 MLA drafter: some draft tokens took the MLA prefill path; "
            "the sliding window is applied to decode-path tokens only.",
            error=False,
        )
    try:
        block_table = decode.block_table
        seq_lens = decode.seq_lens
        tot_seq_lens = decode.dcp_tot_seq_lens if cp_size > 1 else seq_lens
    except AttributeError as e:
        _df2_window_warn_once(
            "fields",
            "DFlash2 MLA drafter: sliding window NOT applied, unexpected decode "
            "metadata layout (%r).",
            e,
        )
        return None
    if tot_seq_lens is None or block_table.shape[0] != seq_lens.shape[0]:
        _df2_window_warn_once(
            "shape",
            "DFlash2 MLA drafter: sliding window NOT applied (dcp_tot_seq_lens=%s, "
            "block_table rows=%d, seq_lens rows=%d).",
            "None" if tot_seq_lens is None else "set",
            block_table.shape[0],
            seq_lens.shape[0],
        )
        return None

    # Uniform K+1 query rows per request (same view the decode kernels take).
    q_len = -(-num_decode_tokens // num_decodes)
    entry_tokens = block_size * cp_size
    excess = (tot_seq_lens.to(torch.int64) - (q_len + window_left)).clamp(min=0)
    skip = excess // entry_tokens  # whole block-table entries to drop
    num_entries = block_table.shape[1]
    index = torch.arange(num_entries, device=block_table.device, dtype=torch.int64)
    index = (index.unsqueeze(0) + skip.unsqueeze(1)).clamp(max=num_entries - 1)

    windowed_decode = copy.copy(decode)
    object.__setattr__(windowed_decode, "block_table", block_table.gather(1, index))
    object.__setattr__(
        windowed_decode,
        "seq_lens",
        seq_lens - (skip * block_size).to(seq_lens.dtype),
    )
    if cp_size > 1:
        object.__setattr__(
            windowed_decode,
            "dcp_tot_seq_lens",
            tot_seq_lens - (skip * entry_tokens).to(tot_seq_lens.dtype),
        )
    windowed = copy.copy(attn_metadata)
    object.__setattr__(windowed, "decode", windowed_decode)
    return windowed


def _install_dflash2_mla_window(attn: nn.Module, window_left: int) -> None:
    """Route ``attn._attention`` through a windowed copy of its decode metadata.

    Instance-level override on DFlash2-MLA draft layers only. The wrapper is the
    eager break point (as the original method is), so under a breakable CUDA
    graph the window is recomputed from the replay-time forward context; under a
    FULL graph or eager it runs inline. The forward-context entry is swapped for
    the duration of the call and restored afterwards.
    """
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture
    from vllm.forward_context import get_forward_context

    method = type(attn)._attention
    inner = getattr(method, "__wrapped__", method)  # strip the eager-break wrapper
    cp_size = int(getattr(attn, "dcp_world_size", 1) or 1)
    layer_name = attn.layer_name

    def _windowed_attention(positions, q, kv_c_normed, k_pe, attn_out):
        by_layer = get_forward_context().attn_metadata
        metadata = by_layer.get(layer_name) if isinstance(by_layer, dict) else None
        windowed = None
        if metadata is not None and attn.kv_cache.dim() >= 2:
            windowed = _window_mla_decode_metadata(
                metadata, window_left, int(attn.kv_cache.size(1)), cp_size
            )
        if windowed is None:
            return inner(attn, positions, q, kv_c_normed, k_pe, attn_out)
        by_layer[layer_name] = windowed
        try:
            return inner(attn, positions, q, kv_c_normed, k_pe, attn_out)
        finally:
            by_layer[layer_name] = metadata

    attn._attention = eager_break_during_capture(_windowed_attention)


class K3DFlash2MLADecoderLayer(K3DSparkDecoderLayer):
    """K3 MLA decoder layer with the DFlash2 attention / MLP block convolutions.

    Placement follows DFlash2Qwen3DecoderLayer: ``prepare`` on the normed input
    (before the fused q_a / kv_a projection, so the query and the block's own
    latent read the convolved input) and ``finish`` on the sublayer output.
    ``finish`` is linear in its input for fixed coefficients, so it is applied to
    the un-reduced row-parallel partial and the all-reduce stays fused into the
    following RMSNorm (exact up to rounding; identical when attention TP is 1).
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        layer_idx: int,
        start_layer_id: int,
        prefix: str,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            config=config,
            layer_idx=layer_idx,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        from vllm.model_executor.models.qwen3_dflash2 import DFlashGroupedConv

        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        dflash_config = getattr(config, "dflash_config", None) or {}
        taps = int(dflash_config.get("conv_kernel_size") or 0)
        self.attention_conv: nn.Module | None = None
        self.mlp_conv: nn.Module | None = None
        self.use_block_conv = taps > 0 and _dflash2_mla_conv_enabled()
        if taps > 0:
            conv_args = dict(
                hidden_size=config.hidden_size,
                taps=taps,
                group_size=int(dflash_config["conv_group_size"]),
                # Query rows per request: the bonus token plus the mask tokens.
                block_size=1 + speculative_config.num_speculative_tokens,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=None,
            )
            layer_prefix = maybe_prefix(prefix, f"layers.{start_layer_id + layer_idx}")
            self.attention_conv = DFlashGroupedConv(
                **conv_args, prefix=maybe_prefix(layer_prefix, "attention_conv")
            )
            self.mlp_conv = DFlashGroupedConv(
                **conv_args, prefix=maybe_prefix(layer_prefix, "mlp_conv")
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            # First layer: hidden_states is the (already reduced) embedding.
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = fused_allreduce_rms_norm(
                hidden_states, residual, self.input_layernorm
            )

        if self.use_block_conv:
            assert self.attention_conv is not None and self.mlp_conv is not None
            hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
            )
            hidden_states = self.attention_conv.finish(hidden_states, coefficients)
            hidden_states, residual = fused_allreduce_rms_norm(
                hidden_states, residual, self.post_attention_layernorm
            )
            hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = self.mlp_conv.finish(hidden_states, coefficients)
            return hidden_states, residual

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states, residual = fused_allreduce_rms_norm(
            hidden_states, residual, self.post_attention_layernorm
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class K3DFlash2MLAModel(K3DSparkModel):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int,
        prefix: str,
    ) -> None:
        # The DSpark constructor builds a markov head (needs markov_rank) and
        # sizes context_proj from num_target_layers; build the DFlash2 set here.
        nn.Module.__init__(self)
        from vllm.compilation.backends import set_model_tag
        from vllm.model_executor.models.qwen3_dflash import _get_dflash_fc_input_size
        from vllm.model_executor.models.qwen3_dflash2 import CandidateSelector

        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.quant_config = get_draft_quant_config(vllm_config)
        dflash_config = getattr(self.config, "dflash_config", None) or {}

        # The frozen target embedding is aliased after the draft checkpoint loads.
        self.embed_tokens: nn.Module | None = None
        self.input_embedding_scale = float(
            dflash_config.get("input_embedding_scale", 1.0)
        )

        # DFlash "fc": concatenated target aux hidden states -> hidden_size. The
        # width is the one the speculator stages (len(aux layers) * hidden).
        self.context_proj = ReplicatedLinear(
            _get_dflash_fc_input_size(vllm_config),
            self.config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "context_proj"),
        )
        # DFlash "hidden_norm".
        self.context_norm = RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )

        self.layers = nn.ModuleList(
            [
                K3DFlash2MLADecoderLayer(
                    vllm_config=vllm_config,
                    config=self.config,
                    layer_idx=layer_idx,
                    start_layer_id=start_layer_id,
                    prefix=prefix,
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        kv_width = self.config.kv_lora_rank + self.config.qk_rope_head_dim
        self.context_kv_proj = MergedColumnParallelLinear(
            self.config.hidden_size,
            [kv_width] * self.config.num_hidden_layers,
            bias=False,
            return_bias=False,
            quant_config=self.quant_config,
            prefix=maybe_prefix(
                prefix,
                f"layers.{start_layer_id}.self_attn.fused_qkv_a_proj",
            ),
            disable_tp=True,
        )
        # DFlash "norm".
        self.final_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.markov_head = None
        self.confidence_head = None

        # Without its own tag the selector shares the draft head's compile cache.
        with set_model_tag("dflash2_candidate_selector"):
            self.candidate_selector = CandidateSelector(
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
                rank=int(dflash_config["selector_rank"]),
                top_k=int(dflash_config["selector_top_k"]),
                params_dtype=vllm_config.model_config.dtype,
                prefix=maybe_prefix(prefix, "candidate_selector"),
            )
        self._max_num_context_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = super().embed_input_ids(input_ids)
        if self.input_embedding_scale != 1.0:
            embeds = embeds * self.input_embedding_scale
        return embeds


class K3DFlash2MLAForCausalLM(K3DSparkForCausalLM):
    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None
    hf_to_vllm_mapper = WeightsMapper(
        # DFlash2 checkpoint names -> the K3 MLA drafter's module names. Regexes
        # run in order on the updated key: an optional leading "model." is
        # dropped, then fc / hidden_norm / norm are renamed.
        orig_to_new_regex={
            re.compile(r"^model\."): "",
            re.compile(r"^fc\."): "context_proj.",
            re.compile(r"^hidden_norm\."): "context_norm.",
            re.compile(r"^norm\."): "final_norm.",
        },
        # The embedding and LM head come from the target.
        orig_to_new_substr={
            "embed_tokens": None,
            "lm_head": None,
            "d2t": None,
            "t2d": None,
        },
        orig_to_new_prefix={"": "model."},
        orig_to_new_stacked={
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
            ".q_a_proj": (".fused_qkv_a_proj", 0),
            ".kv_a_proj_with_mqa": (".fused_qkv_a_proj", 1),
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        dflash_config = getattr(self.config, "dflash_config", None) or {}
        target_layer_num = vllm_config.model_config.get_total_num_hidden_layers()
        self.model = K3DFlash2MLAModel(
            vllm_config=vllm_config,
            start_layer_id=target_layer_num,
            prefix=maybe_prefix(prefix, "model"),
        )

        # Assigned from the target by load_dflash_model (has_own_lm_head=False).
        self.lm_head: nn.Module | None = None
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        draft_vocab_size = (
            getattr(self.config, "draft_vocab_size", None) or self.config.vocab_size
        )
        self.logits_processor = LogitsProcessor(draft_vocab_size, scale=logit_scale)
        softcap = float(dflash_config.get("final_logit_softcapping") or 0.0)
        # Candidates index the selector codebooks, which have the draft's
        # vocabulary; the target's vocabulary can be larger.
        self.candidate_logits_processor = LogitsProcessor(
            min(vllm_config.model_config.get_vocab_size(), self.config.vocab_size),
            scale=float(dflash_config.get("output_multiplier", 1.0)),
            soft_cap=softcap if softcap > 0 else None,
        )

        # The draft's latent cache shares the target's DCP layout; the DFlash
        # speculator keeps DCP on the draft only when draft use_mla is true
        # (hf_config_override marks this draft MLA). Fail fast on a mismatch.
        draft_dcp = vllm_config.parallel_config.decode_context_parallel_size
        target_parallel_config = getattr(
            vllm_config.speculative_config, "target_parallel_config", None
        )
        target_dcp = getattr(
            target_parallel_config, "decode_context_parallel_size", draft_dcp
        )
        if draft_dcp != target_dcp:
            raise ValueError(
                f"DFlash2 MLA drafter: draft DCP={draft_dcp} but target "
                f"DCP={target_dcp}. The draft must be marked MLA "
                "(draft_model_config.use_mla); check that the speculative.py "
                "overlay is mounted, or run with DCP=1."
            )

        causal = self.get_draft_attn_causal()
        conv_on = all(layer.use_block_conv for layer in self.model.layers)
        _df2_logger.info(
            "DFlash2 MLA drafter (K3): %d layers, fc in=%d, block convs %s "
            "(kernel=%s, group=%s, block=%d), selector rank=%s top_k=%s, "
            "mask_token_id=%s, block-causal=%s, dcp=%d",
            len(self.model.layers),
            self.model.context_proj.input_size,
            "on" if conv_on else "OFF",
            dflash_config.get("conv_kernel_size"),
            dflash_config.get("conv_group_size"),
            1 + vllm_config.speculative_config.num_speculative_tokens,
            dflash_config.get("selector_rank"),
            dflash_config.get("selector_top_k"),
            dflash_config.get("mask_token_id"),
            all(causal),
            draft_dcp,
        )
        # The checkpoint is trained with a sliding window on every layer. The K3
        # MLA layer / MLAAttentionSpec have no window, so it is applied here as
        # a per-request KV start in the draft's decode metadata (page-rounded
        # superset, as TRT's _window_mla_block_fixup).
        sliding_window = getattr(self.config, "sliding_window", None)
        if getattr(self.config, "use_sliding_window", False) and sliding_window:
            layer_types = getattr(self.config, "layer_types", None)
            if layer_types and any(t != "sliding_attention" for t in layer_types):
                raise NotImplementedError(
                    "DFlash2 MLA drafter: mixed sliding/full layer_types are not "
                    f"supported ({layer_types})."
                )
            window_left = int(sliding_window) - 1
            backend = self.model.layers[0].self_attn.attn_backend.get_name()
            if not _dflash2_mla_window_enabled():
                _df2_logger.warning(
                    "DFlash2 MLA drafter: the checkpoint's sliding_window=%d is "
                    "NOT applied (VLLM_DFLASH2_MLA_WINDOW=0); draft attention "
                    "reads the whole context.",
                    sliding_window,
                )
            elif backend not in _DFLASH2_MLA_WINDOW_BACKENDS:
                _df2_logger.error(
                    "DFlash2 MLA drafter: the checkpoint's sliding_window=%d is "
                    "NOT applied: draft attention backend %s is not supported "
                    "for the window (supported: %s); draft attention reads the "
                    "whole context.",
                    sliding_window,
                    backend,
                    ", ".join(sorted(_DFLASH2_MLA_WINDOW_BACKENDS)),
                )
            else:
                for layer in self.model.layers:
                    _install_dflash2_mla_window(layer.self_attn, window_left)
                _df2_logger.info(
                    "DFlash2 MLA drafter: sliding_window=%d applied as a draft "
                    "attention window (window_left=%d context tokens before the "
                    "block; start rounded down to one block-table entry = "
                    "kv block_size x dcp(%d) tokens; backend %s).",
                    sliding_window,
                    window_left,
                    draft_dcp,
                    backend,
                )
        else:
            _df2_logger.warning(
                "DFlash2 MLA drafter: no sliding window in the draft config "
                "(use_sliding_window=%s, sliding_window=%s); draft attention "
                "reads the whole context.",
                getattr(self.config, "use_sliding_window", None),
                sliding_window,
            )
        if dflash_config.get("aux_hidden_stream") == "attn_res":
            from vllm import envs

            if not envs.VLLM_KIMI_K3_AUX_ATTN_RES_STREAM:
                _df2_logger.warning(
                    "DFlash2 MLA drafter was trained on the AttnRes aux stream "
                    "(dflash_config.aux_hidden_stream=attn_res) but "
                    "VLLM_KIMI_K3_AUX_ATTN_RES_STREAM is off; the target will "
                    "hand it prefix-only aux states and acceptance will drop. "
                    "Set VLLM_KIMI_K3_AUX_ATTN_RES_STREAM=1."
                )

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # As in DFlashQwen3ForCausalLM: the target's aux streams may not be in
        # the draft's dtype (no-op when they already are).
        return self.model.combine_hidden_states(
            hidden_states.to(self.model.context_proj.params_dtype)
        )

    def get_draft_attn_causal(self) -> list[bool]:
        """Per-layer attention causality, aligned with
        get_draft_kv_cache_layer_names (block-causal: is_causal=true)."""
        from vllm.model_executor.models.qwen3_dflash import _dflash_layer_causal

        return [
            _dflash_layer_causal(self.config, i) for i in range(len(self.model.layers))
        ]

    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.lm_head is not None
        return self.candidate_logits_processor.get_top_k_tokens(
            self.lm_head, hidden_states, self.model.candidate_selector.top_k
        )
