# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for public text-task inference helpers."""

from types import SimpleNamespace

import torch

from alpamayo2_super.models.token_utils import extract_text_tokens
from alpamayo2_super.text_tasks import (
    COT_AUTO_LABELING_KEYS,
    DEFAULT_GROUNDING_QUESTION,
    build_text_task_messages,
    generate_text,
    parse_auto_labeling_json,
    prepare_vqa_inputs,
    summarize_auto_labeling_conditioning,
    split_cot_and_meta_action,
)


def test_default_grounding_question_matches_training_style() -> None:
    """The public grounding demo uses the concise JSON request seen during training."""
    assert DEFAULT_GROUNDING_QUESTION == (
        "Find the white lead vehicle directly ahead and return its bounding box in JSON."
    )


def _sample_data() -> dict:
    """Build a tiny one-sample PhysicalAI-style payload."""
    return {
        "image_frames": torch.zeros((2, 4, 3, 4, 4), dtype=torch.uint8),
        "camera_indices": torch.tensor([0, 1], dtype=torch.int64),
        "absolute_timestamps": torch.tensor(
            [
                [4_800_000, 4_900_000, 5_000_000, 5_100_000],
                [4_800_000, 4_900_000, 5_000_000, 5_100_000],
            ],
            dtype=torch.int64,
        ),
        "ego_t0": torch.tensor([5_100_000], dtype=torch.int64),
        "ego_t0_frame_idx": torch.tensor([3], dtype=torch.int64),
        "ego_history_tvals": torch.tensor([-0.1, 0.0], dtype=torch.float32),
        "ego_history_xyz": torch.zeros((1, 1, 2, 3), dtype=torch.float32),
        "ego_history_rot": torch.eye(3).view(1, 1, 1, 3, 3).repeat(1, 1, 2, 1, 1),
        "ego_future_tvals": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
        "ego_future_xyz": torch.zeros((1, 1, 3, 3), dtype=torch.float32),
        "ego_future_rot": torch.eye(3).view(1, 1, 1, 3, 3).repeat(1, 1, 3, 1, 1),
    }


def _model_config() -> SimpleNamespace:
    """Build the minimal model config surface used by the helper."""
    return SimpleNamespace(
        tokens_per_history_traj=2,
        tokens_per_future_traj=3,
        include_camera_ids=False,
        frame_label="frame_num",
    )


def _joined_text(messages: list[dict]) -> str:
    return "\n".join(
        entry["text"]
        for message in messages
        for entry in message["content"]
        if entry["type"] == "text"
    )


def test_meta_action_messages_match_no_special_eval_prompt() -> None:
    """Meta-action inference asks for cot, meta_action, then future trajectory."""
    messages = build_text_task_messages(
        data=_sample_data(),
        model_config=_model_config(),
        task="meta_action",
    )

    assert [message["role"] for message in messages] == ["system", "user"]
    text = _joined_text(messages)
    assert (
        "output the chain-of-thought reasoning of the driving process, then "
        "output meta actions, then output the future trajectory."
    ) in text
    assert "<|traj_history_start|>" in text
    assert "<|traj_future_start|>" not in text
    assert "<|cot_start|>" not in text
    assert "<|meta_action_start|>" not in text


def test_auto_labeling_messages_condition_on_future_trajectory() -> None:
    """Auto-labeling inference mirrors the no-special future-conditioned JSON task."""
    messages = build_text_task_messages(
        data=_sample_data(),
        model_config=_model_config(),
        task="auto_labeling",
    )

    assert [message["role"] for message in messages] == ["system", "user"]
    text = _joined_text(messages)
    assert "output a comprehensive analysis of the driving scene in JSON format." in text
    assert "<|traj_history_start|>" in text
    assert "<|traj_future_start|>" in text
    assert "<|traj_future_end|>" in text
    assert "<|cot_auto_labeling" not in text


def test_auto_labeling_conditioning_matches_training_temporal_contract() -> None:
    """The public task uses context video through t0 plus a future ego trajectory."""
    summary = summarize_auto_labeling_conditioning(_sample_data())

    assert summary == {
        "camera_input": "context_through_t0",
        "camera_frame_count": 4,
        "future_camera_frames_consumed": False,
        "ego_t0_frame_idx": 3,
        "camera_time_offsets_s": [-0.3, -0.2, -0.1, 0.0],
        "history_trajectory_steps": 2,
        "future_trajectory_steps": 3,
        "future_trajectory_time_range_s": [0.1, 0.3],
    }


def test_auto_labeling_conditioning_rejects_future_camera_frames() -> None:
    """Future video belongs to the offline teacher, not the A2S auto-labeling input."""
    data = _sample_data()
    data["image_frames"] = torch.zeros((2, 5, 3, 4, 4), dtype=torch.uint8)
    data["absolute_timestamps"] = torch.tensor(
        [
            [4_800_000, 4_900_000, 5_000_000, 5_100_000, 5_200_000],
            [4_800_000, 4_900_000, 5_000_000, 5_100_000, 5_200_000],
        ],
        dtype=torch.int64,
    )

    try:
        summarize_auto_labeling_conditioning(data)
    except ValueError as exc:
        assert "future camera frames" in str(exc)
    else:
        raise AssertionError("expected post-t0 camera frames to be rejected")


def test_vqa_messages_match_no_special_eval_prompt() -> None:
    """VQA inference mirrors Alpax no-special eval: images plus plain question only."""
    data = _sample_data()
    data["question"] = "What traffic elements should influence ego behavior?"

    messages = build_text_task_messages(
        data=data,
        model_config=_model_config(),
        task="vqa",
    )

    assert [message["role"] for message in messages] == ["system", "user"]
    text = _joined_text(messages)
    assert "What traffic elements should influence ego behavior?" in text
    assert "<|traj_history_start|>" not in text
    assert "<|traj_future_start|>" not in text
    assert "<|question_start|>" not in text
    assert "<|answer_start|>" not in text
    assert "output the future trajectory" not in text


def test_prepare_vqa_inputs_requires_question() -> None:
    """The VQA convenience helper should make question passing explicit."""
    try:
        prepare_vqa_inputs(
            data=_sample_data(),
            model_config=_model_config(),
            tokenizer=object(),
            question="",
        )
    except ValueError as exc:
        assert "question" in str(exc)
    else:
        raise AssertionError("expected empty VQA question to be rejected")


def test_generate_text_vqa_decodes_only_generated_tokens_without_traj_fusion() -> None:
    """VQA uses the HF text path but should not require trajectory token fusion."""

    class FakeTokenizer:
        pad_token_id = 0

        def batch_decode(self, output_tokens, skip_special_tokens: bool = False):  # noqa: ANN001
            del skip_special_tokens
            assert output_tokens.tolist() == [[42, 43]]
            return ["The lane is partially blocked by construction equipment.<|im_end|>"]

    class FakeVLM:
        generation_config = SimpleNamespace()

        def generate(self, **kwargs):  # noqa: ANN001
            assert kwargs["input_ids"].tolist() == [[1, 2, 3]]
            return SimpleNamespace(sequences=torch.tensor([[1, 2, 3, 42, 43]]))

    model = SimpleNamespace(
        config=SimpleNamespace(
            traj_ids={"history_id0": 100, "future_id0": 200},
            traj_vocab_size=10,
        ),
        tokenizer=FakeTokenizer(),
        vlm=FakeVLM(),
    )

    decoded = generate_text(
        model,
        {
            "task": "vqa",
            "tokenized_data": {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            },
        },
        max_new_tokens=16,
    )

    assert decoded["answer"] == ["The lane is partially blocked by construction equipment."]


def test_generate_text_auto_labeling_fuses_future_trajectory(monkeypatch) -> None:
    """The future XYZ and rotation tensors reach trajectory-token fusion."""
    captured = {}
    history_tokenizer = object()
    future_tokenizer = object()

    def fake_fuse_traj_tokens(
        seen_history_tokenizer,  # noqa: ANN001
        seen_future_tokenizer,  # noqa: ANN001
        input_ids,  # noqa: ANN001
        traj_data,  # noqa: ANN001
        traj_ids,  # noqa: ANN001
    ):
        captured["history_tokenizer"] = seen_history_tokenizer
        captured["future_tokenizer"] = seen_future_tokenizer
        captured["input_ids"] = input_ids
        captured["traj_data"] = traj_data
        captured["traj_ids"] = traj_ids
        return torch.tensor([[7, 8, 9]], dtype=torch.long)

    monkeypatch.setattr(
        "alpamayo2_super.text_tasks.fuse_traj_tokens",
        fake_fuse_traj_tokens,
    )

    class FakeTokenizer:
        pad_token_id = 0

        def batch_decode(self, output_tokens, skip_special_tokens: bool = False):  # noqa: ANN001
            del output_tokens, skip_special_tokens
            return [
                '{"critical_components_analysis": null, '
                '"ego_vehicle_motion_analysis": null, '
                '"trajectory_analysis": "Ego follows the supplied trajectory.", '
                '"chain_of_causation": "The supplied future determines the label."}'
                "<|im_end|>"
            ]

    class FakeVLM:
        generation_config = SimpleNamespace()

        def generate(self, **kwargs):  # noqa: ANN001
            assert kwargs["input_ids"].tolist() == [[7, 8, 9]]
            return SimpleNamespace(sequences=torch.tensor([[7, 8, 9, 42]]))

    sample = _sample_data()
    traj_ids = {"history_id0": 100, "future_id0": 200}
    model = SimpleNamespace(
        config=SimpleNamespace(traj_ids=traj_ids, traj_vocab_size=10),
        tokenizer=FakeTokenizer(),
        history_traj_tokenizer=history_tokenizer,
        future_traj_tokenizer=future_tokenizer,
        vlm=FakeVLM(),
    )

    decoded = generate_text(
        model,
        {
            "task": "auto_labeling",
            "tokenized_data": {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            },
            "ego_history_xyz": sample["ego_history_xyz"],
            "ego_history_rot": sample["ego_history_rot"],
            "ego_future_xyz": sample["ego_future_xyz"],
            "ego_future_rot": sample["ego_future_rot"],
        },
        max_new_tokens=16,
    )

    assert captured["history_tokenizer"] is history_tokenizer
    assert captured["future_tokenizer"] is future_tokenizer
    assert captured["traj_ids"] == traj_ids
    assert captured["traj_data"]["ego_future_xyz"] is sample["ego_future_xyz"]
    assert captured["traj_data"]["ego_future_rot"] is sample["ego_future_rot"]
    assert decoded["cot_auto_labeling_json"][0]["trajectory_analysis"] == (
        "Ego follows the supplied trajectory."
    )


def test_split_cot_and_meta_action_uses_first_axis_marker() -> None:
    """No-special meta-action generations split at the first axis marker."""
    cot, meta_action = split_cot_and_meta_action(
        "The lead vehicle is slowing, so the ego should keep distance.\n"
        "Longitudinal: Keep Speed.\n"
        "Lateral: Go Straight.\n"
        "Lane: Lane Keep."
    )

    assert cot == "The lead vehicle is slowing, so the ego should keep distance."
    assert meta_action.startswith("Longitudinal: Keep Speed.")
    assert "Lane: Lane Keep." in meta_action


def test_extract_text_tokens_splits_no_special_meta_action() -> None:
    """Decoded model outputs expose clean cot and meta_action fields."""

    class FakeTokenizer:
        def batch_decode(self, output_tokens, skip_special_tokens: bool = False):  # noqa: ANN001
            del output_tokens, skip_special_tokens
            return [
                "<|im_start|>assistant\n"
                "Slow for the lead vehicle.\n"
                "Longitudinal: Strong Deceleration.\n"
                "Lateral: Go Straight.\n"
                "Lane: Lane Keep."
                "<|traj_future_start|>"
            ]

    decoded = extract_text_tokens(FakeTokenizer(), torch.empty((1, 0), dtype=torch.long))

    assert decoded["cot"] == ["Slow for the lead vehicle."]
    assert decoded["meta_action"] == [
        "Longitudinal: Strong Deceleration.\nLateral: Go Straight.\nLane: Lane Keep."
    ]
    assert decoded["answer"] == ["Slow for the lead vehicle."]
    assert decoded["box"] == [""]


def test_extract_text_tokens_exposes_no_special_grounding_box() -> None:
    """No-special grounding JSON is exposed through both answer and box fields."""

    class FakeTokenizer:
        def batch_decode(self, output_tokens, skip_special_tokens: bool = False):  # noqa: ANN001
            del output_tokens, skip_special_tokens
            return [
                '<|im_start|>assistant\n[{"bbox_2d": [10, 20, 30, 40], "label": "lead car"}]'
                "<|im_end|>"
            ]

    decoded = extract_text_tokens(FakeTokenizer(), torch.empty((1, 0), dtype=torch.long))

    assert decoded["answer"] == ['[{"bbox_2d": [10, 20, 30, 40], "label": "lead car"}]']
    assert decoded["box"] == ['[{"bbox_2d": [10, 20, 30, 40], "label": "lead car"}]']


def test_parse_auto_labeling_json_normalizes_schema() -> None:
    """Auto-labeling JSON parsing returns the fixed train-time schema."""
    parsed = parse_auto_labeling_json(
        "Here is the label:\n"
        '{"chain_of_causation": "Ego follows the lane.", '
        '"critical_components_analysis": ""}'
    )

    assert tuple(parsed) == COT_AUTO_LABELING_KEYS
    assert parsed["critical_components_analysis"] is None
    assert parsed["ego_vehicle_motion_analysis"] is None
    assert parsed["trajectory_analysis"] is None
    assert parsed["chain_of_causation"] == "Ego follows the lane."
