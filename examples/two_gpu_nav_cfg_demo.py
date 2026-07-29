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

"""Advanced two-GPU navigation classifier-free guidance demo.

This example keeps the default release inference API unchanged and demonstrates the
manual placement needed for navigation CFG on two 80GB GPUs:

- VLM generation and guided/unguided prompt prefill run on ``cuda:0``.
- The expert denoiser, expert masks, and guided/unguided KV caches run on ``cuda:1``.

Generic Hugging Face ``device_map="auto"`` or ``device_map="balanced"`` sharding is
intentionally not used here because the guided and unguided caches must move together
before expert denoising.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import torch

from alpamayo2_super.common.constants import PUBLIC_MODEL_ID


os.environ.setdefault("MPLBACKEND", "Agg")

DEFAULT_MODEL_ID = PUBLIC_MODEL_ID
DEFAULT_NAV_TEXT = (
    "Proceed forward while keeping extra clearance from construction equipment in the right lane."
)
DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parent / "validation_samples.json"


def _move_tensor_container_to_device(value: Any, device: torch.device | str) -> Any:
    """Move tensors inside simple containers without changing non-tensor values."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [_move_tensor_container_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensor_container_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_tensor_container_to_device(item, device) for key, item in value.items()}
    return value


def first_parameter_device(module: torch.nn.Module) -> torch.device:
    """Return the first parameter or buffer device for ``module``."""
    try:
        return next(module.parameters()).device
    except StopIteration:
        try:
            return next(module.buffers()).device
        except StopIteration:
            return torch.device("cpu")


def move_cache_to_device(cache: Any, device: torch.device | str) -> Any:
    """Move tensor attributes inside a Transformers DynamicCache to ``device``."""
    for layer in getattr(cache, "layers", []):
        for attr, value in list(vars(layer).items()):
            setattr(layer, attr, _move_tensor_container_to_device(value, device))
    return cache


def move_plain_tensor_attrs_to_device(module: torch.nn.Module, device: torch.device | str) -> None:
    """Move unregistered tensor attributes that ``nn.Module.to`` does not see."""
    for child in module.modules():
        for attr, value in list(vars(child).items()):
            if attr in {"_parameters", "_buffers", "_modules"}:
                continue
            setattr(child, attr, _move_tensor_container_to_device(value, device))


def _validate_cuda_devices(
    vlm_device: str,
    expert_device: str,
) -> tuple[torch.device, torch.device]:
    """Validate the explicit two-GPU placement requested by the demo."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The two-GPU navigation CFG demo requires CUDA. "
            "Check nvidia-smi and CUDA_VISIBLE_DEVICES."
        )
    vlm = torch.device(vlm_device)
    expert = torch.device(expert_device)
    if vlm.type != "cuda" or expert.type != "cuda":
        raise ValueError("--vlm-device and --expert-device must both be CUDA devices.")
    if vlm.index is None or expert.index is None:
        raise ValueError("Pass explicit CUDA device indexes, for example cuda:0 and cuda:1.")
    if vlm.index == expert.index:
        raise ValueError("Navigation CFG needs separate VLM and expert CUDA devices.")
    device_count = torch.cuda.device_count()
    if vlm.index >= device_count or expert.index >= device_count:
        raise ValueError(
            f"Requested {vlm} and {expert}, but only {device_count} CUDA devices are visible."
        )
    return vlm, expert


def prepare_nav_model_inputs(data: dict[str, Any], model: Any, nav_text: str) -> dict[str, Any]:
    """Tokenize guided and unguided prompts for one PhysicalAI-AV sample."""
    from alpamayo2_super.chat_template.conversation import build_conversation
    from alpamayo2_super.helper import get_processor

    data_for_nav = dict(data)
    data_for_nav["nav_text"] = [nav_text]
    processor = get_processor(model.tokenizer, model.config)
    images = data_for_nav["image_frames"].flatten(0, 1)
    images = (images.float() / 255.0) if images.dtype == torch.uint8 else images.float()

    def tokenize_prompt(components_order: list[str]) -> dict[str, Any]:
        messages = build_conversation(
            data=data_for_nav,
            num_tokens_per_history_traj=model.config.tokens_per_history_traj,
            num_tokens_per_future_traj=model.config.tokens_per_future_traj,
            components_order=components_order,
            components_prompt=["cot", "traj_future"],
            generation_mode=True,
            include_camera_ids=model.config.include_camera_ids,
            camera_ids=data_for_nav["camera_indices"],
            include_frame_nums=model.config.frame_label == "frame_num",
        )
        if messages[-1]["role"] == "assistant" and not messages[-1]["content"]:
            messages = messages[:-1]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=False,
        )
        return dict(
            processor(
                text=text,
                images=images,
                videos=None,
                padding=False,
                return_tensors="pt",
                do_rescale=False,
            )
        )

    tokenized_data = tokenize_prompt(["image", "traj_history", "nav_instruction", "prompt"])
    unguided_tokenized_data = tokenize_prompt(["image", "traj_history", "prompt"])
    unguided_tokenized_data.pop("pixel_values", None)
    unguided_tokenized_data.pop("image_grid_thw", None)
    if (
        tokenized_data["input_ids"].shape[0] != 1
        or unguided_tokenized_data["input_ids"].shape[0] != 1
    ):
        raise ValueError("prepare_nav_model_inputs expects one sample at a time")

    return {
        "tokenized_data": tokenized_data,
        "unguided_tokenized_data": unguided_tokenized_data,
        "ego_history_xyz": data_for_nav["ego_history_xyz"],
        "ego_history_rot": data_for_nav["ego_history_rot"],
    }


def build_unguided_prefill_inputs(
    tokenized_data: dict[str, Any],
    unguided_tokenized_data: dict[str, Any],
    pad_token_id: int,
) -> dict[str, Any]:
    """Combine unguided text inputs with the shared guided visual payload."""
    unguided_input_ids = unguided_tokenized_data["input_ids"]
    unguided_prefix_mask = unguided_tokenized_data.get("attention_mask")
    if unguided_prefix_mask is None:
        unguided_prefix_mask = unguided_input_ids.ne(pad_token_id).long()
    return {
        "input_ids": unguided_input_ids,
        "attention_mask": unguided_prefix_mask,
        "image_grid_thw": tokenized_data.get("image_grid_thw"),
        "pixel_values": tokenized_data.get("pixel_values"),
    }


def build_unguided_continuation_inputs(
    guided_sequences: torch.Tensor,
    guided_prefix_length: int,
    unguided_input_ids: torch.Tensor,
    unguided_prefix_mask: torch.Tensor,
    pad_token_id: int,
    num_traj_samples: int,
) -> dict[str, torch.Tensor]:
    """Build inputs that replay the guided generation after the unguided prefix."""
    guided_generated_tokens = guided_sequences[:, guided_prefix_length:]
    generated_mask = guided_generated_tokens.ne(pad_token_id).long()
    repeated_prefix_mask = unguided_prefix_mask.repeat_interleave(
        num_traj_samples,
        dim=0,
    )
    attention_mask = torch.cat([repeated_prefix_mask, generated_mask], dim=1)
    cache_position = torch.arange(
        unguided_input_ids.shape[1],
        unguided_input_ids.shape[1] + guided_generated_tokens.shape[1],
        device=unguided_input_ids.device,
        dtype=torch.long,
    )
    full_sequences = torch.cat(
        [
            unguided_input_ids.repeat_interleave(num_traj_samples, dim=0),
            guided_generated_tokens,
        ],
        dim=1,
    )
    return {
        "input_ids": guided_generated_tokens,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "full_sequences": full_sequences,
    }


def load_model_on_two_gpus(model_id: str, vlm_device: str, expert_device: str) -> Any:
    """Load Alpamayo 2 Super and place VLM/expert components manually."""
    from alpamayo2_super.inference_smoke import validate_model_id
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    validate_model_id(model_id)
    model = Alpamayo2Super.from_pretrained(model_id, dtype=torch.bfloat16)
    model.vlm.to(vlm_device)
    if hasattr(model.history_traj_tokenizer, "to"):
        model.history_traj_tokenizer.to(vlm_device)
    if hasattr(model.future_traj_tokenizer, "to"):
        model.future_traj_tokenizer.to(vlm_device)
    model.expert.to(expert_device)
    move_plain_tensor_attrs_to_device(model.expert, expert_device)
    model.eval()
    return model


def _empty_cuda_cache() -> None:
    """Release cached CUDA blocks after large VLM/expert phases."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def sample_with_nav_cfg(
    model: Any,
    model_inputs: dict[str, Any],
    diffusion_steps: int = 10,
    top_p: float = 0.98,
    temperature: float = 0.6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any], float]:
    """Sample one trajectory with guided/unguided navigation CFG."""
    import einops
    import numpy as np
    from transformers import StoppingCriteriaList
    from transformers.generation.logits_process import LogitsProcessorList

    from alpamayo2_super.models.alpamayo2_super import (
        MaskDiscreteTrajectoryLogitsProcessor,
        _append_text_eos_mask,
    )
    from alpamayo2_super.models.expert_utils import (
        StopAfterEOS,
        build_expert_pos_ids_and_attn_mask,
        find_eos_offset,
        replace_padding_after_eos,
    )
    from alpamayo2_super.models.token_utils import extract_text_tokens
    from alpamayo2_super.models.utils import fuse_traj_tokens

    tokenized_data = dict(model_inputs["tokenized_data"])
    unguided_tokenized_data = dict(model_inputs["unguided_tokenized_data"])
    traj_data = {
        "ego_history_xyz": model_inputs["ego_history_xyz"],
        "ego_history_rot": model_inputs["ego_history_rot"],
    }
    input_ids = fuse_traj_tokens(
        model.history_traj_tokenizer,
        model.future_traj_tokenizer,
        tokenized_data["input_ids"],
        traj_data,
        model.config.traj_ids,
    )
    unguided_input_ids = fuse_traj_tokens(
        model.history_traj_tokenizer,
        model.future_traj_tokenizer,
        unguided_tokenized_data["input_ids"],
        traj_data,
        model.config.traj_ids,
    )
    tokenized_data["input_ids"] = input_ids
    unguided_tokenized_data["input_ids"] = unguided_input_ids
    num_traj_samples = 1

    generation_config = copy.deepcopy(model.vlm.generation_config)
    generation_config.top_p = top_p
    generation_config.temperature = temperature
    generation_config.do_sample = True
    generation_config.num_return_sequences = num_traj_samples
    generation_config.max_new_tokens = max(256, model.config.tokens_per_future_traj)
    generation_config.output_logits = False
    generation_config.return_dict_in_generate = True
    generation_config.top_k = None
    generation_config.pad_token_id = model.tokenizer.pad_token_id

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
    eos_token_id = model.config.traj_ids["future_start"]
    _append_text_eos_mask(
        logits_processor,
        generation_config.eos_token_id,
        preserved_token_id=eos_token_id,
    )
    stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
    vlm_outputs = model.vlm.generate(
        **tokenized_data,
        generation_config=generation_config,
        stopping_criteria=stopping_criteria,
        logits_processor=logits_processor,
    )
    if hasattr(vlm_outputs, "logits"):
        del vlm_outputs.logits
    _empty_cuda_cache()

    vlm_outputs.rope_deltas = model.vlm.model.rope_deltas
    vlm_outputs.sequences = replace_padding_after_eos(
        token_ids=vlm_outputs.sequences,
        eos_token_id=eos_token_id,
        pad_token_id=model.tokenizer.pad_token_id,
    )
    guided_prompt_cache = vlm_outputs.past_key_values
    guided_prefill_seq_len = guided_prompt_cache.get_seq_length()
    vlm_device = input_ids.device
    expert_device = first_parameter_device(model.expert)
    batch_size = vlm_outputs.sequences.shape[0]
    num_diffusion_tokens = model.expert.action_space.get_action_space_dims()[0]

    guided_offset = find_eos_offset(
        sequences=vlm_outputs.sequences,
        eos_token_id=eos_token_id,
        device=vlm_device,
    )
    guided_prefix_mask = tokenized_data.get("attention_mask")
    if guided_prefix_mask is not None:
        guided_prefix_mask = torch.repeat_interleave(
            guided_prefix_mask,
            num_traj_samples,
            dim=0,
        )
    guided_position_ids, guided_attention_mask = build_expert_pos_ids_and_attn_mask(
        offset=guided_offset,
        rope_deltas=vlm_outputs.rope_deltas,
        kv_cache_seq_len=guided_prefill_seq_len,
        n_diffusion_tokens=num_diffusion_tokens,
        b_star=batch_size,
        device=vlm_device,
        prefix_mask=guided_prefix_mask,
    )

    unguided_prefill_inputs = build_unguided_prefill_inputs(
        tokenized_data=tokenized_data,
        unguided_tokenized_data=unguided_tokenized_data,
        pad_token_id=model.tokenizer.pad_token_id,
    )
    unguided_prefix_mask = unguided_prefill_inputs["attention_mask"]
    unguided_prefill_outputs = model.vlm(
        **unguided_prefill_inputs,
        use_cache=True,
        logits_to_keep=1,
    )
    unguided_prompt_cache = unguided_prefill_outputs.past_key_values
    del unguided_prefill_outputs
    _empty_cuda_cache()
    unguided_prompt_cache.batch_repeat_interleave(num_traj_samples)

    unguided_continuation_inputs = build_unguided_continuation_inputs(
        guided_sequences=vlm_outputs.sequences,
        guided_prefix_length=tokenized_data["input_ids"].shape[1],
        unguided_input_ids=unguided_input_ids,
        unguided_prefix_mask=unguided_prefix_mask,
        pad_token_id=model.tokenizer.pad_token_id,
        num_traj_samples=num_traj_samples,
    )
    guided_generated_tokens = unguided_continuation_inputs["input_ids"]
    # Expert CFG removes navigation conditioning while retaining the guided CoT.
    unguided_vlm_outputs = model.vlm(
        input_ids=guided_generated_tokens,
        attention_mask=unguided_continuation_inputs["attention_mask"],
        past_key_values=unguided_prompt_cache,
        cache_position=unguided_continuation_inputs["cache_position"],
        use_cache=True,
        logits_to_keep=1,
    )
    unguided_prompt_cache = unguided_vlm_outputs.past_key_values
    unguided_rope_deltas = getattr(unguided_vlm_outputs, "rope_deltas", model.vlm.model.rope_deltas)
    if hasattr(unguided_vlm_outputs, "logits"):
        del unguided_vlm_outputs.logits
    _empty_cuda_cache()

    unguided_offset = find_eos_offset(
        sequences=unguided_continuation_inputs["full_sequences"],
        eos_token_id=eos_token_id,
        device=vlm_device,
        warn=False,
    )
    unguided_prefix_mask_repeated = torch.repeat_interleave(
        unguided_prefix_mask,
        num_traj_samples,
        dim=0,
    )
    unguided_position_ids, unguided_attention_mask = build_expert_pos_ids_and_attn_mask(
        offset=unguided_offset,
        rope_deltas=unguided_rope_deltas,
        kv_cache_seq_len=unguided_prompt_cache.get_seq_length(),
        n_diffusion_tokens=num_diffusion_tokens,
        b_star=batch_size,
        device=vlm_device,
        prefix_mask=unguided_prefix_mask_repeated,
    )

    if expert_device != vlm_device:
        guided_prompt_cache = move_cache_to_device(guided_prompt_cache, expert_device)
        unguided_prompt_cache = move_cache_to_device(unguided_prompt_cache, expert_device)
        guided_position_ids = guided_position_ids.to(expert_device)
        guided_attention_mask = guided_attention_mask.to(expert_device)
        unguided_position_ids = unguided_position_ids.to(expert_device)
        unguided_attention_mask = unguided_attention_mask.to(expert_device)

    forward_kwargs = {}
    if model.expert.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    def expert_step(
        action: torch.Tensor,
        timestep: torch.Tensor,
        past_key_values: Any,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run one expert denoising step against a specific prompt cache."""
        future_token_embeds = model.expert.action_in_proj(action, timestep)
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(
                batch_size,
                num_diffusion_tokens,
                -1,
            )
        cache_len = past_key_values.get_seq_length()
        expert_outputs = model.expert.expert(
            inputs_embeds=future_token_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=True,
            **forward_kwargs,
        )
        past_key_values.crop(cache_len)
        pred = model.expert.action_out_proj(expert_outputs.last_hidden_state)
        return pred.view(-1, *model.expert.action_space.get_action_space_dims())

    def guided_step_fn(*, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Run one guided expert denoising step."""
        return expert_step(
            x,
            t,
            guided_prompt_cache,
            guided_position_ids,
            guided_attention_mask,
        )

    def unguided_step_fn(*, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Run one unguided expert denoising step."""
        return expert_step(
            x,
            t,
            unguided_prompt_cache,
            unguided_position_ids,
            unguided_attention_mask,
        )

    guidance_weight = float(getattr(model.expert.diffusion, "inference_guidance_weight", 1.0))
    action = model.expert.diffusion.sample(
        batch_size=batch_size,
        step_fn=guided_step_fn,
        unguided_step_fn=unguided_step_fn,
        device=expert_device,
        return_all_steps=False,
        inference_step=diffusion_steps,
        use_classifier_free_guidance=True,
        inference_guidance_weight=guidance_weight,
    )

    hist_xyz = traj_data["ego_history_xyz"][:, -1].to(expert_device)
    hist_rot = traj_data["ego_history_rot"][:, -1].to(expert_device)
    hist_xyz_rep = einops.repeat(hist_xyz, "b ... -> (b n) ...", n=num_traj_samples)
    hist_rot_rep = einops.repeat(hist_rot, "b ... -> (b n) ...", n=num_traj_samples)
    pred_xyz, pred_rot = model.expert.action_space.action_to_traj(
        action,
        hist_xyz_rep,
        hist_rot_rep,
    )
    pred_xyz = einops.rearrange(pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=1, nj=1)
    pred_rot = einops.rearrange(pred_rot, "(b ns nj) ... -> b ns nj ...", ns=1, nj=1)
    logprob = torch.zeros_like(pred_xyz[..., 0])

    extra = extract_text_tokens(model.tokenizer, guided_generated_tokens)
    for key in extra:
        extra[key] = np.array(extra[key]).reshape([1, 1, 1])
    return pred_xyz, pred_rot, logprob, extra, guidance_weight


def _first_cot(extra: dict[str, Any]) -> str:
    """Return the first decoded CoT string from an ``extra`` dict."""
    import numpy as np

    if "cot" not in extra:
        return ""
    cot = np.asarray(extra["cot"], dtype=object)
    return "" if cot.size == 0 else str(cot.reshape(-1)[0])


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload with parent directories created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_two_gpu_nav_cfg(
    model_id: str,
    clip_id: str,
    t0_us: int,
    nav_text: str = DEFAULT_NAV_TEXT,
    vlm_device: str = "cuda:0",
    expert_device: str = "cuda:1",
    save_viz: str | None = None,
    save_json: str | None = None,
    diffusion_steps: int = 10,
    seed: int = 42,
    top_p: float = 0.98,
    temperature: float = 0.6,
    require_camera_projection: bool = False,
) -> None:
    """Load one PhysicalAI-AV sample and run manual two-GPU navigation CFG."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    vlm, expert = _validate_cuda_devices(vlm_device, expert_device)

    from alpamayo2_super import helper
    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo2_super.visualization import plot_inference_result

    print(f"Loading dataset for clip_id: {clip_id}...")
    data = load_physical_aiavdataset(
        clip_id,
        t0_us=t0_us,
    )
    print("Dataset loaded.")

    print(f"Loading model with manual placement: VLM={vlm}, expert={expert}...")
    model = load_model_on_two_gpus(model_id, str(vlm), str(expert))
    model_inputs = prepare_nav_model_inputs(data, model, nav_text)
    model_inputs = helper.to_device(model_inputs, vlm)

    torch.cuda.manual_seed_all(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, logprob, extra, guidance_weight = sample_with_nav_cfg(
            model=model,
            model_inputs=model_inputs,
            diffusion_steps=diffusion_steps,
            top_p=top_p,
            temperature=temperature,
        )
    del pred_rot, logprob

    cot = _first_cot(extra)
    print("Navigation instruction:", nav_text)
    print("CFG guidance weight:", guidance_weight)
    print("Chain-of-Causation:\n", cot)

    if save_viz or save_json:
        import matplotlib.pyplot as plt

        figure, metadata = plot_inference_result(
            data=data,
            pred_xyz=pred_xyz,
            extra=extra,
            output_path=save_viz,
            json_path=None,
            model_id=model_id,
            seed=seed,
            require_camera_projection=require_camera_projection,
        )
        plt.close(figure)
        metadata.update(
            {
                "nav_text": nav_text,
                "use_classifier_free_guidance_nav": True,
                "nav_guidance_weight": guidance_weight,
                "device_placement": {
                    "vlm_device": str(vlm),
                    "expert_device": str(expert),
                },
                "diffusion_steps": diffusion_steps,
                "top_p": top_p,
                "temperature": temperature,
                "dtype": "bfloat16",
            }
        )
        if save_viz:
            print("Saved visualization:", save_viz)
        if save_json:
            _write_json(save_json, metadata)
            print("Saved metadata:", save_json)
        if metadata["min_ade_m"] is not None:
            print("minADE:", metadata["min_ade_m"], "meters")


def main() -> None:
    """Run the two-GPU navigation CFG demo."""
    from alpamayo2_super.inference_smoke import load_validation_sample

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default=os.environ.get("ALPAMAYO2_SUPER_MODEL_ID", DEFAULT_MODEL_ID),
        help="Hugging Face repo ID or local release-native checkpoint path.",
    )
    parser.add_argument(
        "--manifest",
        default=os.environ.get("ALPAMAYO2_SUPER_VALIDATION_MANIFEST", str(DEFAULT_MANIFEST_PATH)),
        help="Optional validation_samples.json manifest. Overrides --clip-id/--t0-us.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=int(os.environ.get("ALPAMAYO2_SUPER_SAMPLE_INDEX", "0")),
        help="Sample index to load from --manifest.",
    )
    parser.add_argument("--clip-id", default=None, help="PhysicalAI-AV clip ID to load.")
    parser.add_argument("--t0-us", default=None, help="Clip timestamp in microseconds.")
    parser.add_argument(
        "--nav-text",
        default=os.environ.get("ALPAMAYO2_SUPER_NAV_TEXT", DEFAULT_NAV_TEXT),
        help="Navigation instruction included in the guided prompt.",
    )
    parser.add_argument("--vlm-device", default="cuda:0", help="CUDA device for VLM generation.")
    parser.add_argument(
        "--expert-device",
        default="cuda:1",
        help="CUDA device for expert denoising.",
    )
    parser.add_argument("--save-viz", default=None, help="Optional PNG output path.")
    parser.add_argument("--save-json", default=None, help="Optional JSON sidecar output path.")
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=int(os.environ.get("ALPAMAYO2_SUPER_DIFFUSION_STEPS", "10")),
        help="Number of expert diffusion integration steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("ALPAMAYO2_SUPER_SEED", "42")),
        help="CUDA random seed.",
    )
    parser.add_argument("--top-p", type=float, default=0.98, help="VLM nucleus sampling value.")
    parser.add_argument("--temperature", type=float, default=0.6, help="VLM sampling temperature.")
    parser.add_argument(
        "--require-camera-projection",
        action="store_true",
        help="Fail if calibration is unavailable or no trajectory projects into camera views.",
    )
    args = parser.parse_args()

    if args.manifest and Path(args.manifest).is_file():
        sample = load_validation_sample(Path(args.manifest), args.sample_index)
        clip_id = sample["clip_id"]
        t0_us = sample["t0_us"]
    else:
        if args.clip_id is None or args.t0_us is None:
            parser.error("Pass --manifest with a valid file, or pass both --clip-id and --t0-us.")
        try:
            clip_id = args.clip_id
            t0_us = int(args.t0_us)
        except ValueError:
            parser.error("--t0-us must be an integer")

    try:
        run_two_gpu_nav_cfg(
            model_id=args.model_id,
            clip_id=clip_id,
            t0_us=t0_us,
            nav_text=args.nav_text,
            vlm_device=args.vlm_device,
            expert_device=args.expert_device,
            save_viz=args.save_viz,
            save_json=args.save_json,
            diffusion_steps=args.diffusion_steps,
            seed=args.seed,
            top_p=args.top_p,
            temperature=args.temperature,
            require_camera_projection=args.require_camera_projection,
        )
    except (FileNotFoundError, NotADirectoryError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
