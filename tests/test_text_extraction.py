# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regressions for text extraction, independent of model runtime dependencies."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOKEN_UTILS_PATH = ROOT / "src/alpamayo2_super/models/token_utils.py"
TEXT_TASKS_PATH = ROOT / "src/alpamayo2_super/text_tasks.py"
TRAJECTORY_START = "<|traj_future_start|>"
AUTO_LABEL = {
    "critical_components_analysis": "Lane: A cyclist occupies the right lane.",
    "ego_vehicle_motion_analysis": "Longitudinal: The ego vehicle is slowing.",
    "trajectory_analysis": "Lateral: The path remains in the current lane.",
    "chain_of_causation": "Yield to the cyclist.",
}


class _TokenBatch:
    """Small token transport supporting the generation wrapper's prompt slicing."""

    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    @classmethod
    def from_text(cls, *texts):
        return cls([list(map(ord, text)) for text in texts])

    def __getitem__(self, key):
        rows, columns = key
        return _TokenBatch([row[columns] for row in self.rows[rows]])


class _Tokenizer:
    pad_token_id = 0

    def batch_decode(self, tokens, *, skip_special_tokens):
        assert skip_special_tokens is False
        return ["".join(map(chr, row)) for row in tokens.rows]


def _install_module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def token_utils(monkeypatch):
    # Only import-time types are stubbed; extraction executes the production module.
    _install_module(
        monkeypatch,
        "torch",
        Tensor=_TokenBatch,
        LongTensor=_TokenBatch,
        FloatTensor=_TokenBatch,
        inference_mode=lambda: lambda function: function,
    )
    _install_module(
        monkeypatch,
        "transformers",
        AutoTokenizer=_Tokenizer,
        StoppingCriteria=object,
        StoppingCriteriaList=list,
    )
    _install_module(
        monkeypatch,
        "alpamayo2_super.models.utils",
        SPECIAL_TOKENS={"traj_future_start": TRAJECTORY_START},
        fuse_traj_tokens=lambda history, future, tokens, data, ids: tokens,
    )
    return _load_module("_test_token_utils", TOKEN_UTILS_PATH)


@pytest.fixture
def text_tasks(monkeypatch, token_utils):
    monkeypatch.setitem(sys.modules, "alpamayo2_super.models.token_utils", token_utils)
    _install_module(
        monkeypatch,
        "transformers.generation.logits_process",
        LogitsProcessorList=list,
    )
    _install_module(
        monkeypatch,
        "alpamayo2_super.chat_template.conversation",
        **{
            name: Mock()
            for name in (
                "construct_image",
                "construct_system_prompt",
                "construct_traj_future",
                "construct_traj_history",
                "construct_user_prompt",
            )
        },
    )
    _install_module(monkeypatch, "alpamayo2_super.config", Alpamayo2SuperConfig=object)
    _install_module(monkeypatch, "alpamayo2_super.helper", get_processor=Mock())
    _install_module(
        monkeypatch,
        "alpamayo2_super.models.expert_utils",
        StopAfterEOS=Mock(),
        replace_padding_after_eos=lambda token_ids, **kwargs: token_ids,
    )
    _install_module(
        monkeypatch,
        "alpamayo2_super.models.alpamayo2_super",
        MaskDiscreteTrajectoryLogitsProcessor=Mock(),
    )
    return _load_module("_test_text_tasks", TEXT_TASKS_PATH)


@pytest.mark.parametrize(
    "text",
    [
        "Lane: The right lane is closed.",
        "Lateral: A pedestrian is crossing from the left.",
        "Longitudinal: The lead vehicle is slowing.",
        "Scene observations:\nLane: The right lane is closed.\nTraffic: A bus is ahead.",
        "1. Lane: Right lane closed.\n2. Lateral: Pedestrian on left.\n3. Longitudinal: Slow car.",
    ],
)
def test_vqa_preserves_complete_answer(token_utils, text):
    result = token_utils.extract_text_tokens(_Tokenizer(), _TokenBatch.from_text(text), task="vqa")

    assert result["answer"] == result["cot"] == result["raw_outputs"] == [text]
    assert result["meta_action"] == [""]


@pytest.mark.parametrize("task", ["vqa", "auto_labeling"])
def test_text_tasks_preserve_literal_trajectory_marker(token_utils, task):
    text = f"The notation {TRAJECTORY_START} is mentioned in this answer."

    result = token_utils.extract_text_tokens(_Tokenizer(), _TokenBatch.from_text(text), task=task)

    assert result["answer"] == [text]


def test_auto_labeling_preserves_json_with_axis_headings(token_utils):
    text = json.dumps(AUTO_LABEL)
    result = token_utils.extract_text_tokens(
        _Tokenizer(), _TokenBatch.from_text(text), task="auto_labeling"
    )

    assert result["cot_auto_labeling"] == result["answer"] == [text]
    assert json.loads(result["cot_auto_labeling"][0]) == AUTO_LABEL
    assert result["meta_action"] == [""]
    assert result["raw_outputs"] == [text]


@pytest.mark.parametrize("task", ["trajectory", "meta_action"])
def test_driving_tasks_keep_cot_and_meta_action_split(token_utils, task):
    meta_action = "Longitudinal: brake\nLateral: straight\nLane: keep"
    text = f"A red light is ahead.\n{meta_action}{TRAJECTORY_START}<i42>"
    tokens = _TokenBatch.from_text(text)

    result = token_utils.extract_text_tokens(_Tokenizer(), tokens, task=task)

    assert result["answer"] == result["cot"] == ["A red light is ahead."]
    assert result["meta_action"] == [meta_action]
    assert result["raw_outputs"] == [text]
    assert result["box"] == result["cot_auto_labeling"] == [""]
    assert token_utils.extract_text_tokens(_Tokenizer(), tokens) == result


@pytest.mark.parametrize("task", ["trajectory", "meta_action", "vqa", "auto_labeling"])
def test_assistant_and_eos_cleanup_preserves_raw_output(token_utils, task):
    raw = (
        "<|im_start|>assistant\nEarlier answer.<|im_end|>"
        "<|im_start|>user\nNext question.<|im_end|>"
        "<|im_start|>assistant\n  A cyclist is ahead.  <|im_end|>padding"
    )

    result = token_utils.extract_text_tokens(_Tokenizer(), _TokenBatch.from_text(raw), task=task)

    assert result["answer"] == ["A cyclist is ahead."]
    assert result["raw_outputs"] == [raw]


@pytest.mark.parametrize("task", ["vqa", "auto_labeling"])
def test_grounding_json_stays_available_in_text_fields(token_utils, task):
    text = '{"label": "Lane: right", "bbox_2d": [1, 2, 3, 4]}'

    result = token_utils.extract_text_tokens(_Tokenizer(), _TokenBatch.from_text(text), task=task)

    assert result["answer"] == result["box"] == result["cot_auto_labeling"] == [text]
    assert result["meta_action"] == [""]


@pytest.mark.parametrize("task", ["vqa", "auto_labeling", "meta_action"])
def test_generate_text_uses_requested_task_and_excludes_prompt(text_tasks, task):
    prompt = "Question with Lane: in the prompt."
    answer = {
        "vqa": "1. Lane: Right lane closed.\n2. Lateral: A cyclist is approaching.",
        "auto_labeling": json.dumps(AUTO_LABEL),
        "meta_action": "Slow for traffic.\nLongitudinal: brake\nLane: keep",
    }[task]
    raw = answer + "<|im_end|>"
    model = SimpleNamespace(
        tokenizer=_Tokenizer(),
        history_traj_tokenizer=None,
        future_traj_tokenizer=None,
        config=SimpleNamespace(
            traj_ids={"history_id0": 1000, "future_id0": 2000, "future_start": 3000},
            traj_vocab_size=1000,
        ),
        vlm=SimpleNamespace(
            generation_config=SimpleNamespace(),
            generate=Mock(
                return_value=SimpleNamespace(sequences=_TokenBatch.from_text(prompt + raw))
            ),
        ),
    )
    data = {
        "task": task,
        "tokenized_data": {"input_ids": _TokenBatch.from_text(prompt)},
        "ego_history_xyz": None,
        "ego_history_rot": None,
        "ego_future_xyz": None,
        "ego_future_rot": None,
    }

    result = text_tasks.generate_text(model, data)

    assert result["raw_outputs"] == [raw]
    if task == "meta_action":
        assert result["answer"] == ["Slow for traffic."]
        assert result["meta_action"] == ["Longitudinal: brake\nLane: keep"]
    else:
        assert result["answer"] == [answer]
        assert result["meta_action"] == [""]
    if task == "auto_labeling":
        assert result["cot_auto_labeling_json"] == [AUTO_LABEL]
