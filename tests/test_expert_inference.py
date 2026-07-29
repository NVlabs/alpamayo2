# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused regression tests for release-native expert inference."""

import inspect
from types import SimpleNamespace

import pytest
import torch
from transformers import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList

from alpamayo2_super.diffusion.flow_matching import FlowMatching
from alpamayo2_super.models.action_in_proj import PerWaypointActionInProjV2
from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super


class _FakeCache:
    """Minimal cache surface used by shared-prefill generation."""

    def __init__(self) -> None:
        self.repeat_count: int | None = None

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Record the requested batch expansion."""
        self.repeat_count = repeats


class _FakeVLMModel(torch.nn.Module):
    """Record a prompt-only VLM prefill."""

    def __init__(self, cache: _FakeCache) -> None:
        super().__init__()
        self.cache = cache
        self.calls: list[dict] = []
        self.rope_deltas = torch.tensor([[10], [20]])

    def forward(self, **kwargs):
        """Record and return one cache."""
        self.calls.append(kwargs)
        return SimpleNamespace(past_key_values=self.cache)


class _FakeVLM(torch.nn.Module):
    """Record the cached generation continuation."""

    def __init__(self) -> None:
        super().__init__()
        self.cache = _FakeCache()
        self.model = _FakeVLMModel(self.cache)
        self.generate_calls: list[dict] = []

    def generate(self, **kwargs):
        """Record generation inputs and return a small output."""
        self.generate_calls.append(kwargs)
        return SimpleNamespace(sequences=torch.zeros((6, 1), dtype=torch.long))


def test_shared_prefill_runs_vision_once_before_expanding_samples() -> None:
    """Trajectory samples should share one prompt/vision prefill."""
    model = object.__new__(Alpamayo2Super)
    torch.nn.Module.__init__(model)
    model.vlm = _FakeVLM()
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    attention_mask = torch.ones_like(input_ids)
    pixel_values = torch.randn(2, 3)
    generation_config = GenerationConfig(
        do_sample=True,
        num_return_sequences=3,
        output_logits=True,
        return_dict_in_generate=True,
    )
    stopping_criteria = object()

    outputs = model._generate_with_shared_prefill(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        },
        generation_config,
        n_samples_total=3,
        stopping_criteria=stopping_criteria,
    )

    assert len(model.vlm.model.calls) == 1
    prefill_call = model.vlm.model.calls[0]
    torch.testing.assert_close(prefill_call["input_ids"], input_ids[:, :-1])
    torch.testing.assert_close(prefill_call["attention_mask"], attention_mask[:, :-1])
    assert prefill_call["pixel_values"] is pixel_values
    assert prefill_call["use_cache"] is True
    assert model.vlm.cache.repeat_count == 3

    generate_call = model.vlm.generate_calls[0]
    torch.testing.assert_close(
        generate_call["input_ids"],
        torch.repeat_interleave(input_ids, 3, dim=0),
    )
    torch.testing.assert_close(
        generate_call["attention_mask"],
        torch.repeat_interleave(attention_mask, 3, dim=0),
    )
    assert "pixel_values" not in generate_call
    assert generate_call["past_key_values"] is model.vlm.cache
    assert generate_call["stopping_criteria"] is stopping_criteria
    assert generation_config.num_return_sequences == 1
    assert generation_config.output_logits is False
    torch.testing.assert_close(
        outputs.rope_deltas,
        torch.tensor([[10], [10], [10], [20], [20], [20]]),
    )


def test_action_projection_matches_bfloat16_weights_without_autocast() -> None:
    """Fourier features should be cast before entering a bf16 projection MLP."""
    projection = PerWaypointActionInProjV2(
        in_dims=[2, 2],
        out_dim=8,
        num_enc_layers=1,
        hidden_size=8,
        num_fourier_feats=4,
    ).to(dtype=torch.bfloat16)

    output = projection(
        torch.randn(2, 3, 2, dtype=torch.float32),
        torch.rand(2, 1, 1, dtype=torch.float32),
    )

    assert output.shape == (2, 3, 8)
    assert output.dtype == torch.bfloat16


def test_text_eos_mask_preserves_expert_anchor() -> None:
    """Expert generation masks text EOS IDs but keeps its trajectory anchor available."""
    from alpamayo2_super.models.alpamayo2_super import (
        MaskTokenIdsLogitsProcessor,
        _append_text_eos_mask,
    )

    processors = LogitsProcessorList()

    _append_text_eos_mask(processors, [2, 7], preserved_token_id=7)

    assert len(processors) == 1
    assert isinstance(processors[0], MaskTokenIdsLogitsProcessor)
    scores = torch.zeros((1, 10))
    masked = processors[0](torch.ones((1, 2), dtype=torch.long), scores)
    assert torch.isneginf(masked[0, 2])
    assert masked[0, 7] == 0


@pytest.mark.parametrize("text_eos_ids", [2, [1, 3]])
def test_text_eos_mask_blocks_scalar_and_list_configs(text_eos_ids: int | list[int]) -> None:
    """Both Hugging Face EOS configuration forms should be masked."""
    from alpamayo2_super.models.alpamayo2_super import _append_text_eos_mask

    processors = LogitsProcessorList()
    _append_text_eos_mask(processors, text_eos_ids)
    scores = torch.arange(10, dtype=torch.float32).reshape(2, 5)

    masked_scores = processors(torch.tensor([[0], [0]]), scores.clone())

    masked_ids = [text_eos_ids] if isinstance(text_eos_ids, int) else text_eos_ids
    kept_ids = [token_id for token_id in range(scores.shape[1]) if token_id not in masked_ids]
    assert torch.isneginf(masked_scores[:, masked_ids]).all()
    torch.testing.assert_close(masked_scores[:, kept_ids], scores[:, kept_ids])


def test_text_eos_mask_ignores_missing_config() -> None:
    """Models without a configured text EOS leave generation unchanged."""
    from alpamayo2_super.models.alpamayo2_super import _append_text_eos_mask

    processors = LogitsProcessorList()
    _append_text_eos_mask(processors, None)

    assert not processors


def test_flow_matching_applies_classifier_free_guidance() -> None:
    """CFG should combine unguided and guided vector fields inside the sampler."""
    diffusion = FlowMatching(
        x_dims=[2],
        num_inference_steps=1,
        use_classifier_free_guidance=True,
        inference_guidance_weight=2.0,
    )

    output = diffusion.sample(
        batch_size=1,
        step_fn=lambda *, x, t: torch.ones_like(x),
        unguided_step_fn=lambda *, x, t: torch.zeros_like(x),
        temperature=0.0,
    )

    torch.testing.assert_close(output, torch.full((1, 2), 2.0))


def test_flow_matching_requires_unguided_step_for_cfg() -> None:
    """CFG cannot silently fall back to the guided field twice."""
    diffusion = FlowMatching(x_dims=[2], use_classifier_free_guidance=True)

    with pytest.raises(ValueError, match="unguided_step_fn is required"):
        diffusion.sample(
            batch_size=1,
            step_fn=lambda *, x, t: torch.ones_like(x),
        )


def test_flow_matching_preserves_device_as_third_positional_argument() -> None:
    """Adding CFG support must not change the established positional sampler API."""
    parameter_names = list(inspect.signature(FlowMatching.sample).parameters)

    assert parameter_names[:4] == ["self", "batch_size", "step_fn", "device"]
