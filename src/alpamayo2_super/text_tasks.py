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

"""Public helpers for Alpamayo 2 Super text-generation tasks."""

import copy
import json
from typing import Any, Literal

import torch
from transformers import StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList

from alpamayo2_super.chat_template.conversation import (
    construct_image,
    construct_system_prompt,
    construct_traj_future,
    construct_traj_history,
    construct_user_prompt,
)
from alpamayo2_super.config import Alpamayo2SuperConfig
from alpamayo2_super.helper import get_processor
from alpamayo2_super.models.expert_utils import StopAfterEOS, replace_padding_after_eos
from alpamayo2_super.models.token_utils import extract_text_tokens, split_cot_and_meta_action
from alpamayo2_super.models.utils import fuse_traj_tokens

TextTask = Literal["meta_action", "auto_labeling", "vqa"]

COT_AUTO_LABELING_KEYS = (
    "critical_components_analysis",
    "ego_vehicle_motion_analysis",
    "trajectory_analysis",
    "chain_of_causation",
)

AUTO_LABELING_CONTEXT_FRAME_COUNT = 4

DEFAULT_GROUNDING_QUESTION = (
    "Find the white lead vehicle directly ahead and return its bounding box in JSON."
)

_TASK_PROMPTS: dict[TextTask, list[str]] = {
    "meta_action": ["cot", "meta_action", "traj_future"],
    "auto_labeling": ["cot_auto_labeling"],
    "vqa": [],
}


def _validate_task(task: str) -> TextTask:
    if task not in _TASK_PROMPTS:
        raise ValueError(f"task must be one of {tuple(_TASK_PROMPTS)}, got {task!r}")
    return task  # type: ignore[return-value]


def _include_frame_nums(model_config: Alpamayo2SuperConfig) -> bool:
    return getattr(model_config, "frame_label", "frame_num") == "frame_num"


def _validate_vqa_question(question: Any) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("VQA question must be a non-empty string")
    return question.strip()


def build_text_task_messages(
    data: dict[str, Any],
    model_config: Alpamayo2SuperConfig,
    task: TextTask,
) -> list[dict[str, Any]]:
    """Build no-special text-generation messages for a release inference task."""
    task = _validate_task(task)
    user_content: list[dict[str, Any]] = []
    user_content.extend(
        construct_image(
            data=data,
            include_camera_ids=model_config.include_camera_ids,
            camera_ids=data["camera_indices"],
            include_frame_nums=_include_frame_nums(model_config),
        )
    )
    if task == "vqa":
        user_content.append({"type": "text", "text": _validate_vqa_question(data.get("question"))})
        return [
            {"role": "system", "content": construct_system_prompt()},
            {"role": "user", "content": user_content},
        ]

    user_content.extend(
        construct_traj_history(
            num_tokens_per_history_traj=model_config.tokens_per_history_traj,
        )
    )
    if task == "auto_labeling":
        user_content.extend(
            construct_traj_future(
                num_tokens_per_future_traj=model_config.tokens_per_future_traj,
            )
        )
    user_content.extend(
        construct_user_prompt(
            components_order=[],
            components_prompt=_TASK_PROMPTS[task],
            generation_mode=True,
        )
    )
    return [
        {"role": "system", "content": construct_system_prompt()},
        {"role": "user", "content": user_content},
    ]


def _normalize_future_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
    """Normalize future tensors to ``[B, n_traj, T, ...]``."""
    if name == "ego_future_xyz":
        if value.ndim == 2:
            return value.unsqueeze(0).unsqueeze(0)
        if value.ndim == 3:
            return value.unsqueeze(1)
        if value.ndim == 4:
            return value
    else:
        if value.ndim == 3:
            return value.unsqueeze(0).unsqueeze(0)
        if value.ndim == 4:
            return value.unsqueeze(1)
        if value.ndim == 5:
            return value
    raise ValueError(f"{name} must have trajectory shape [T, ...], [B, T, ...], or [B, n, T, ...]")


def summarize_auto_labeling_conditioning(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and summarize the train-time auto-labeling conditioning contract.

    The A2S ``cot_auto_labeling`` task consumes four camera context frames ending
    at ``t0``, ego trajectory history, and a future ego trajectory. Future camera
    frames were inputs to the offline teacher that produced the labels, but they
    were not inputs to this A2S training task.
    """
    image_frames = data.get("image_frames")
    if not isinstance(image_frames, torch.Tensor) or image_frames.ndim != 5:
        raise ValueError("image_frames must have shape [camera, frame, channel, height, width]")
    num_cameras, num_frames = image_frames.shape[:2]

    camera_indices = data.get("camera_indices")
    if not isinstance(camera_indices, torch.Tensor) or camera_indices.numel() != num_cameras:
        raise ValueError("camera_indices must contain one id per camera")

    ego_t0_frame_idx = data.get("ego_t0_frame_idx")
    if not isinstance(ego_t0_frame_idx, torch.Tensor) or ego_t0_frame_idx.numel() != 1:
        raise ValueError("ego_t0_frame_idx is required for auto-labeling input validation")
    t0_frame_idx = int(ego_t0_frame_idx.reshape(-1)[0].item())
    if t0_frame_idx != num_frames - 1:
        raise ValueError(
            "A2S auto-labeling does not consume future camera frames; "
            "ego_t0_frame_idx must identify the final context frame"
        )
    if num_frames != AUTO_LABELING_CONTEXT_FRAME_COUNT:
        raise ValueError(
            f"A2S auto-labeling expects {AUTO_LABELING_CONTEXT_FRAME_COUNT} context frames, "
            f"got {num_frames}"
        )

    absolute_timestamps = data.get("absolute_timestamps")
    ego_t0 = data.get("ego_t0")
    if not isinstance(absolute_timestamps, torch.Tensor) or tuple(absolute_timestamps.shape) != (
        num_cameras,
        num_frames,
    ):
        raise ValueError("absolute_timestamps must have shape [camera, frame]")
    if not isinstance(ego_t0, torch.Tensor) or ego_t0.numel() != 1:
        raise ValueError("ego_t0 is required for auto-labeling input validation")
    camera_offsets = (absolute_timestamps.to(torch.float64) - ego_t0.item()) * 1e-6
    if torch.any(camera_offsets[:, 1:] < camera_offsets[:, :-1]):
        raise ValueError("camera timestamps must be monotonic within each camera")
    if float(camera_offsets.max().item()) > 0.05:
        raise ValueError("A2S auto-labeling does not consume future camera frames after t0")
    median_camera_offsets = torch.median(camera_offsets, dim=0).values

    trajectory_specs = (
        ("history", "ego_history_xyz", "ego_history_rot", "ego_history_tvals"),
        ("future", "ego_future_xyz", "ego_future_rot", "ego_future_tvals"),
    )
    trajectory_steps: dict[str, int] = {}
    trajectory_tvals: dict[str, torch.Tensor] = {}
    for name, xyz_key, rot_key, tvals_key in trajectory_specs:
        xyz = data.get(xyz_key)
        rot = data.get(rot_key)
        tvals = data.get(tvals_key)
        if not isinstance(xyz, torch.Tensor) or xyz.ndim != 4 or xyz.shape[-1] != 3:
            raise ValueError(f"{xyz_key} must have shape [batch, trajectory, step, 3]")
        if not isinstance(rot, torch.Tensor) or rot.ndim != 5 or rot.shape[-2:] != (3, 3):
            raise ValueError(f"{rot_key} must have shape [batch, trajectory, step, 3, 3]")
        if xyz.shape[:3] != rot.shape[:3]:
            raise ValueError(f"{xyz_key} and {rot_key} must have matching trajectory shapes")
        if not isinstance(tvals, torch.Tensor) or tvals.ndim != 1 or tvals.numel() != xyz.shape[-2]:
            raise ValueError(f"{tvals_key} must contain one timestamp per trajectory step")
        trajectory_steps[name] = xyz.shape[-2]
        trajectory_tvals[name] = tvals

    history_tvals = trajectory_tvals["history"]
    future_tvals = trajectory_tvals["future"]
    if torch.any(history_tvals > 1e-6) or not torch.isclose(
        history_tvals[-1],
        torch.zeros((), dtype=history_tvals.dtype, device=history_tvals.device),
        atol=1e-6,
    ):
        raise ValueError("ego_history_tvals must end at t0 and contain no future steps")
    if torch.any(future_tvals <= 0):
        raise ValueError("ego_future_tvals must contain only post-t0 trajectory steps")

    return {
        "camera_input": "context_through_t0",
        "camera_frame_count": num_frames,
        "future_camera_frames_consumed": False,
        "ego_t0_frame_idx": t0_frame_idx,
        "camera_time_offsets_s": [round(float(value), 6) for value in median_camera_offsets],
        "history_trajectory_steps": trajectory_steps["history"],
        "future_trajectory_steps": trajectory_steps["future"],
        "future_trajectory_time_range_s": [
            round(float(future_tvals[0].item()), 6),
            round(float(future_tvals[-1].item()), 6),
        ],
    }


def prepare_text_generation_inputs(
    data: dict[str, Any],
    model_config: Alpamayo2SuperConfig,
    tokenizer: Any,
    task: TextTask,
    future_xyz: torch.Tensor | None = None,
    future_rot: torch.Tensor | None = None,
    question: str | None = None,
) -> dict[str, Any]:
    """Tokenize one PhysicalAI-AV sample for a public text-generation task.

    Auto-labeling follows the training/eval template that conditions on four
    context camera frames through ``t0`` and a future ego trajectory. By default
    that trajectory is ``data["ego_future_*"]``; pass ``future_xyz`` and
    ``future_rot`` to condition on a model-predicted future instead. Future camera
    frames are not part of this A2S task's input contract.
    """
    task = _validate_task(task)
    if (future_xyz is None) != (future_rot is None):
        raise ValueError("future_xyz and future_rot must be provided together")

    data_for_task = dict(data)
    if question is not None:
        data_for_task["question"] = _validate_vqa_question(question)
    if future_xyz is not None and future_rot is not None:
        data_for_task["ego_future_xyz"] = _normalize_future_tensor("ego_future_xyz", future_xyz)
        data_for_task["ego_future_rot"] = _normalize_future_tensor("ego_future_rot", future_rot)

    if task == "auto_labeling":
        for key in ("ego_future_xyz", "ego_future_rot"):
            if key not in data_for_task:
                raise KeyError(f"{key} is required for auto_labeling text generation")
        summarize_auto_labeling_conditioning(data_for_task)

    processor = get_processor(tokenizer, model_config)
    messages = build_text_task_messages(data_for_task, model_config, task)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        add_vision_id=False,
    )
    images = data_for_task["image_frames"].flatten(0, 1)
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
        raise ValueError("prepare_text_generation_inputs expects one sample at a time")

    model_inputs: dict[str, Any] = {
        "task": task,
        "tokenized_data": tokenized_data,
    }
    if task != "vqa":
        model_inputs["ego_history_xyz"] = data_for_task["ego_history_xyz"]
        model_inputs["ego_history_rot"] = data_for_task["ego_history_rot"]
    if task == "auto_labeling":
        model_inputs["ego_future_xyz"] = data_for_task["ego_future_xyz"]
        model_inputs["ego_future_rot"] = data_for_task["ego_future_rot"]
    return model_inputs


def prepare_vqa_inputs(
    data: dict[str, Any],
    model_config: Alpamayo2SuperConfig,
    tokenizer: Any,
    question: str,
) -> dict[str, Any]:
    """Tokenize one PhysicalAI-AV sample for no-special VQA generation."""
    return prepare_text_generation_inputs(
        data=data,
        model_config=model_config,
        tokenizer=tokenizer,
        task="vqa",
        question=question,
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for start_idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[start_idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("auto-labeling output did not contain a JSON object")


def parse_auto_labeling_json(text: str) -> dict[str, str | None]:
    """Parse and normalize a generated auto-labeling JSON string."""
    payload = _extract_json_object(text)
    normalized: dict[str, str | None] = {}
    for key in COT_AUTO_LABELING_KEYS:
        value = payload.get(key)
        if value is None:
            normalized[key] = None
            continue
        value = value.strip() if isinstance(value, str) else str(value)
        normalized[key] = value or None
    return normalized


@torch.inference_mode()
def generate_text(
    model: Any,
    data: dict[str, Any],
    top_p: float = 0.98,
    top_k: int | None = None,
    temperature: float = 0.6,
    num_samples: int = 1,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    """Generate public text-task outputs from prepared model inputs."""
    task = _validate_task(data["task"])
    tokenized_data = dict(data["tokenized_data"])
    if task != "vqa":
        traj_data = {
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        }
        if task == "auto_labeling":
            traj_data["ego_future_xyz"] = data["ego_future_xyz"]
            traj_data["ego_future_rot"] = data["ego_future_rot"]

        tokenized_data["input_ids"] = fuse_traj_tokens(
            model.history_traj_tokenizer,
            model.future_traj_tokenizer,
            tokenized_data["input_ids"],
            traj_data,
            model.config.traj_ids,
        )
    prompt_length = tokenized_data["input_ids"].shape[1]

    generation_config = copy.deepcopy(model.vlm.generation_config)
    generation_config.top_p = top_p
    generation_config.temperature = temperature
    generation_config.do_sample = True
    generation_config.num_return_sequences = num_samples
    default_max_new_tokens = 1024 if task in ("auto_labeling", "vqa") else 512
    generation_config.max_new_tokens = max_new_tokens or default_max_new_tokens
    generation_config.output_logits = False
    generation_config.return_dict_in_generate = True
    generation_config.top_k = top_k
    generation_config.pad_token_id = model.tokenizer.pad_token_id

    from alpamayo2_super.models.alpamayo2_super import MaskDiscreteTrajectoryLogitsProcessor

    logits_processor = LogitsProcessorList(
        [
            MaskDiscreteTrajectoryLogitsProcessor(
                traj_token_offset=min(
                    model.config.traj_ids["history_id0"],
                    model.config.traj_ids["future_id0"],
                ),
                traj_vocab_size=model.config.traj_vocab_size,
            )
        ]
    )
    generate_kwargs: dict[str, Any] = {
        "generation_config": generation_config,
        "logits_processor": logits_processor,
    }
    if task == "meta_action":
        eos_token_id = model.config.traj_ids["future_start"]
        generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [StopAfterEOS(eos_token_id=eos_token_id)]
        )

    outputs = model.vlm.generate(**tokenized_data, **generate_kwargs)
    sequences = outputs.sequences
    if task == "meta_action":
        sequences = replace_padding_after_eos(
            token_ids=sequences,
            eos_token_id=model.config.traj_ids["future_start"],
            pad_token_id=model.tokenizer.pad_token_id,
        )

    generated_tokens = sequences[:, prompt_length:]
    extracted = extract_text_tokens(model.tokenizer, generated_tokens)
    if task == "auto_labeling":
        auto_labeling_text = [
            text or cot for text, cot in zip(extracted["cot_auto_labeling"], extracted["cot"])
        ]
        extracted["cot_auto_labeling"] = auto_labeling_text
        extracted["cot_auto_labeling_json"] = [
            parse_auto_labeling_json(text) for text in auto_labeling_text
        ]
    return extracted


__all__ = [
    "AUTO_LABELING_CONTEXT_FRAME_COUNT",
    "COT_AUTO_LABELING_KEYS",
    "DEFAULT_GROUNDING_QUESTION",
    "TextTask",
    "build_text_task_messages",
    "generate_text",
    "parse_auto_labeling_json",
    "prepare_text_generation_inputs",
    "prepare_vqa_inputs",
    "split_cot_and_meta_action",
    "summarize_auto_labeling_conditioning",
]
