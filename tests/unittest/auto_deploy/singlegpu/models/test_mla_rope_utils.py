# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.auto_deploy.custom_ops.mla.trtllm_mla import _TrtllmMLAPlanner
from tensorrt_llm._torch.auto_deploy.models.custom.mla_rope_utils import (
    _rope_deinterleave_load_hook,
)
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek import (
    DeepSeekV3YarnRotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v2 import (
    DeepSeekV2YarnRotaryEmbedding,
    _q_proj_deinterleave_hook,
)
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_glm4_moe_lite import (
    Glm4MoeLiteYarnRotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_kimi_k2 import KimiK2YarnRotaryEmbedding
from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig
from tensorrt_llm._torch.auto_deploy.transform.library.fuse_rope_mla import (
    _compute_rotary_cos_sin_from_config,
    _undo_rope_deinterleave,
)


def _flatten_cos_sin(cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return torch.stack([cos.float().cpu(), sin.float().cpu()], dim=-1).reshape(1, -1)


class _Factory:
    def __init__(self, config):
        self.config = config

    def _get_model_config(self):
        return self.config, None


class _DirectQProjModel(nn.Module):
    """Minimal DeepSeek-V2 parameter hierarchy for direct-Q RoPE tests."""

    def __init__(self, hidden_size: int, q_out_features: int):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].self_attn = nn.Module()
        self.model.layers[0].self_attn.q_proj = nn.Linear(hidden_size, q_out_features, bias=False)


@pytest.mark.parametrize(
    "rope_cls",
    [
        DeepSeekV2YarnRotaryEmbedding,
        DeepSeekV3YarnRotaryEmbedding,
        Glm4MoeLiteYarnRotaryEmbedding,
        KimiK2YarnRotaryEmbedding,
    ],
)
@pytest.mark.parametrize("rope_type_key", ["type", "rope_type"])
def test_fused_mla_yarn_config_fallback_matches_model_rope_variants(rope_cls, rope_type_key):
    dim = 64
    max_pos = 128
    rope_scaling = {
        rope_type_key: "yarn",
        "factor": 40.0,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    }
    config = SimpleNamespace(
        rope_theta=10000.0,
        qk_rope_head_dim=dim,
        max_position_embeddings=max_pos,
        rope_scaling=rope_scaling,
    )

    rope = rope_cls(
        dim,
        max_pos,
        base=config.rope_theta,
        scaling_factor=rope_scaling["factor"],
        original_max_position_embeddings=rope_scaling["original_max_position_embeddings"],
        beta_fast=rope_scaling["beta_fast"],
        beta_slow=rope_scaling["beta_slow"],
        mscale=rope_scaling["mscale"],
        mscale_all_dim=rope_scaling["mscale_all_dim"],
    )
    if hasattr(rope, "_ad_cos_cached"):
        expected = _flatten_cos_sin(rope._ad_cos_cached, rope._ad_sin_cached)
    else:
        x = torch.empty(1, 1, 1, dim, dtype=torch.float32, device="cuda")
        cos, sin = rope.to("cuda")(x)
        expected = _flatten_cos_sin(cos, sin)

    actual = _compute_rotary_cos_sin_from_config(_Factory(config)).cpu()
    torch.testing.assert_close(actual, expected, atol=3e-7, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float8_e4m3fn])
def test_rope_deinterleave_load_hook_reorders_q_and_kv_weights(dtype):
    num_heads = 2
    qk_nope_head_dim = 2
    qk_rope_head_dim = 4
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    q_lora_rank = 3
    kv_lora_rank = 2

    q_weight = torch.arange(num_heads * qk_head_dim * q_lora_rank, dtype=torch.float32).reshape(
        num_heads * qk_head_dim, q_lora_rank
    )
    kv_weight = torch.arange(
        (kv_lora_rank + qk_rope_head_dim) * q_lora_rank, dtype=torch.float32
    ).reshape(kv_lora_rank + qk_rope_head_dim, q_lora_rank)
    kv_bias = torch.arange(kv_lora_rank + qk_rope_head_dim, dtype=torch.float32)

    state_dict = {
        "model.layers.0.self_attn.q_b_proj.weight": q_weight.to(dtype),
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight": kv_weight.to(dtype),
        "model.layers.0.self_attn.kv_a_proj_with_mqa.bias": kv_bias.to(dtype),
    }

    _rope_deinterleave_load_hook(
        state_dict,
        prefix="",
        qk_rope_head_dim=qk_rope_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        num_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        num_layers=1,
    )

    perm = torch.tensor([0, 2, 1, 3], dtype=torch.long)
    q_weight_expected = q_weight.to(dtype).float().view(num_heads, qk_head_dim, q_lora_rank)
    q_weight_expected = torch.cat(
        [
            q_weight_expected[:, :qk_nope_head_dim, :],
            q_weight_expected[:, qk_nope_head_dim:, :].index_select(1, perm),
        ],
        dim=1,
    ).reshape(-1, q_lora_rank)
    kv_weight_expected = torch.cat(
        [
            kv_weight.to(dtype).float()[:kv_lora_rank, :],
            kv_weight.to(dtype).float()[kv_lora_rank:, :].index_select(0, perm),
        ],
        dim=0,
    )
    kv_bias_expected = torch.cat(
        [
            kv_bias.to(dtype).float()[:kv_lora_rank],
            kv_bias.to(dtype).float()[kv_lora_rank:].index_select(0, perm),
        ]
    )

    torch.testing.assert_close(
        state_dict["model.layers.0.self_attn.q_b_proj.weight"].float(), q_weight_expected
    )
    torch.testing.assert_close(
        state_dict["model.layers.0.self_attn.kv_a_proj_with_mqa.weight"].float(),
        kv_weight_expected,
    )
    torch.testing.assert_close(
        state_dict["model.layers.0.self_attn.kv_a_proj_with_mqa.bias"].float(),
        kv_bias_expected,
    )


def test_fused_mla_restores_direct_q_proj_to_gptj_layout():
    """DeepSeek-Coder-V2-Lite uses q_proj because q_lora_rank is None."""
    num_heads = 2
    qk_nope_head_dim = 2
    qk_rope_head_dim = 4
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    hidden_size = 3
    original = torch.arange(num_heads * qk_head_dim * hidden_size, dtype=torch.float32).reshape(
        num_heads * qk_head_dim, hidden_size
    )

    state_dict = {"model.layers.0.self_attn.q_proj.weight": original.clone()}
    _q_proj_deinterleave_hook(
        state_dict,
        prefix="",
        qk_rope_head_dim=qk_rope_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        num_heads=num_heads,
        num_layers=1,
    )

    model = _DirectQProjModel(hidden_size, num_heads * qk_head_dim)
    model.load_state_dict(state_dict)
    config = SimpleNamespace(
        qk_rope_head_dim=qk_rope_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=2,
        num_attention_heads=num_heads,
    )

    modified = _undo_rope_deinterleave(
        model,
        _Factory(config),
        SharedConfig(local_rank=0, world_size=1),
    )

    assert modified == 1
    torch.testing.assert_close(model.model.layers[0].self_attn.q_proj.weight, original)


def test_no_fuse_mla_identity_rope_covers_configured_context(monkeypatch):
    qk_rope_head_dim = 4
    max_seq_len = 65536

    def _fake_rope_params(_self):
        inv_freq = torch.ones(qk_rope_head_dim // 2)
        fallback = torch.zeros(1, 8192 * qk_rope_head_dim * 2)
        return inv_freq, fallback

    monkeypatch.setattr(
        "tensorrt_llm._torch.auto_deploy.custom_ops.mla.trtllm_mla."
        "RopeParams.create_rope_const_params",
        _fake_rope_params,
    )
    planner = _TrtllmMLAPlanner()

    planner.ensure_rope_tables(qk_rope_head_dim, max_seq_len)

    assert planner._identity_rope_positions == max_seq_len
    assert planner.identity_cos_sin is not None
    identity = planner.identity_cos_sin.view(max_seq_len, qk_rope_head_dim, 2)
    torch.testing.assert_close(identity[..., 0], torch.ones_like(identity[..., 0]))
    torch.testing.assert_close(identity[..., 1], torch.zeros_like(identity[..., 1]))


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="requires float8 support")
def test_rope_deinterleave_load_hook_preserves_fp8_bytes():
    source = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32).to(torch.float8_e4m3fn)
    state_dict = {"model.layers.0.self_attn.kv_a_proj_with_mqa.bias": source.clone()}

    _rope_deinterleave_load_hook(
        state_dict,
        prefix="",
        qk_rope_head_dim=4,
        qk_nope_head_dim=0,
        num_heads=1,
        kv_lora_rank=0,
        num_layers=1,
    )

    expected = (
        source.view(torch.uint8).index_select(0, torch.tensor([0, 2, 1, 3])).view(source.dtype)
    )
    assert torch.equal(
        state_dict["model.layers.0.self_attn.kv_a_proj_with_mqa.bias"].view(torch.uint8),
        expected.view(torch.uint8),
    )
