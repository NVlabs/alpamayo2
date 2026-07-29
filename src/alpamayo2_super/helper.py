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

"""Helpers for Alpamayo 2 Super inference examples."""

import collections.abc
from typing import Any

import torch
from transformers import AutoProcessor, AutoTokenizer

from alpamayo2_super.chat_template.conversation import build_conversation
from alpamayo2_super.config import Alpamayo2SuperConfig, resolve_checkpoint_name_or_path


def create_messages(
    data: dict[str, Any],
    model_config: Alpamayo2SuperConfig,
) -> list[dict[str, Any]]:
    """Create generation-mode chat messages for CoT + trajectory inference."""
    messages = build_conversation(
        data=data,
        num_tokens_per_history_traj=model_config.tokens_per_history_traj,
        num_tokens_per_future_traj=model_config.tokens_per_future_traj,
        components_order=["image", "traj_history", "prompt"],
        components_prompt=["cot", "traj_future"],
        generation_mode=True,
        include_camera_ids=model_config.include_camera_ids,
        camera_ids=data["camera_indices"],
        include_frame_nums=model_config.frame_label == "frame_num",
    )
    if messages[-1]["role"] == "assistant" and not messages[-1]["content"]:
        messages = messages[:-1]
    return messages


def get_processor(
    tokenizer: AutoTokenizer,
    model_config: Alpamayo2SuperConfig,
) -> AutoProcessor:
    """Load the checkpoint processor and attach the Alpamayo tokenizer."""
    name_or_path = resolve_checkpoint_name_or_path(model_config)
    if name_or_path is None:
        raise ValueError("model_config must include a checkpoint path or vlm_name_or_path")
    processor_kwargs = {"fix_mistral_regex": True}
    if model_config.min_pixels is not None:
        processor_kwargs["min_pixels"] = model_config.min_pixels
    if model_config.max_pixels is not None:
        processor_kwargs["max_pixels"] = model_config.max_pixels

    processor = AutoProcessor.from_pretrained(name_or_path, **processor_kwargs)
    processor.tokenizer = tokenizer
    return processor


def prepare_model_inputs(
    data: dict[str, Any],
    model_config: Alpamayo2SuperConfig,
    tokenizer: AutoTokenizer,
) -> dict[str, Any]:
    """Tokenize one PhysicalAI-AV sample for ``Alpamayo2Super.sample_trajectories_from_data``."""
    processor = get_processor(tokenizer, model_config)
    messages = create_messages(data, model_config)
    has_assistant_content = messages[-1]["role"] == "assistant" and bool(messages[-1]["content"])
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not has_assistant_content,
        add_vision_id=False,
        continue_final_message=has_assistant_content,
    )
    images = data["image_frames"].flatten(0, 1)
    images = (images.float() / 255.0) if images.dtype == torch.uint8 else images.float()
    tokenized_data = dict(
        processor(
            text=text,
            images=images,
            videos=None,
            padding=False,
            return_tensors="pt",
            do_rescale=False,
        )
    )
    if tokenized_data["input_ids"].shape[0] != 1:
        raise ValueError("prepare_model_inputs expects one sample at a time")
    return {
        "tokenized_data": tokenized_data,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }


def to_device(
    data: Any,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Any:
    """Recursively move tensors to ``device`` and optional ``dtype``."""
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    if isinstance(data, collections.abc.Mapping):
        return {key: to_device(value, device=device, dtype=dtype) for key, value in data.items()}
    if isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes)):
        return [to_device(elem, device=device, dtype=dtype) for elem in data]
    return data
