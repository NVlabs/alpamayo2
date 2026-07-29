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

"""Alpamayo 2 Super Hugging Face model wrapper."""

import copy
from dataclasses import dataclass
from typing import Any

import einops
import hydra.utils as hyu
import numpy as np
import torch
import transformers
from transformers import AutoModel, GenerationConfig, PreTrainedModel, StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.utils import ModelOutput

from alpamayo2_super.config import (
    Alpamayo2SuperConfig,
    build_alpamayo2_super_tokenizer,
    resolve_checkpoint_name_or_path,
)
from alpamayo2_super.models.expert import ExpertModel
from alpamayo2_super.models.expert_utils import (
    StopAfterEOS,
    build_expert_pos_ids_and_attn_mask,
    find_eos_offset,
    replace_padding_after_eos,
)
from alpamayo2_super.models.token_utils import extract_text_tokens
from alpamayo2_super.models.utils import IGNORE_INDEX, compute_next_token_loss, fuse_traj_tokens


@dataclass
class Alpamayo2SuperModelOutput(ModelOutput):
    """Training output for Alpamayo 2 Super."""

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None


class MaskDiscreteTrajectoryLogitsProcessor(LogitsProcessor):
    """Mask discrete trajectory token logits during CoT generation."""

    def __init__(self, traj_token_offset: int, traj_vocab_size: int) -> None:
        super().__init__()
        self.traj_token_offset = traj_token_offset
        self.traj_vocab_size = traj_vocab_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Set the discrete trajectory-token logit span to ``-inf``."""
        del input_ids
        scores[:, self.traj_token_offset : self.traj_token_offset + self.traj_vocab_size] = float(
            "-inf"
        )
        return scores


class MaskTokenIdsLogitsProcessor(LogitsProcessor):
    """Mask configured token IDs during generation."""

    def __init__(self, token_ids: list[int]) -> None:
        super().__init__()
        self.token_ids = list(token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Set the configured token logits to ``-inf``."""
        del input_ids
        scores[:, self.token_ids] = float("-inf")
        return scores


def _append_text_eos_mask(
    logits_processor: LogitsProcessorList,
    text_eos_ids: int | list[int] | None,
    preserved_token_id: int | None = None,
) -> None:
    """Prevent text EOS termination without masking the expert anchor."""
    if isinstance(text_eos_ids, int):
        text_eos_ids = [text_eos_ids]
    if text_eos_ids:
        text_eos_ids = [token_id for token_id in text_eos_ids if token_id != preserved_token_id]
    if text_eos_ids:
        logits_processor.append(MaskTokenIdsLogitsProcessor(text_eos_ids))


class Alpamayo2Super(PreTrainedModel):
    """Expert VLA model used by the Alpamayo 2 Super 34B release."""

    config_class = Alpamayo2SuperConfig
    base_model_prefix = "vlm"
    _supports_flash_attn = True
    _supports_sdpa = True
    _keys_to_ignore_on_load_missing = [
        r"expert\.action_in_proj\.sinus\.\d+\.freqs",
        r"expert\.action_in_proj\.timestep_fourier_encoder\.freqs",
        r"expert\.action_space\.(accel_mean|accel_std|curvature_mean|curvature_std)",
    ]

    def __init__(self, config: Alpamayo2SuperConfig) -> None:
        config.set_dtype(config.dtype)
        super().__init__(config)

        vlm_cls = getattr(transformers, config.vlm_class)
        self.vlm = vlm_cls._from_config(config.vlm_config)
        tokenizer_name_or_path = resolve_checkpoint_name_or_path(config)
        if tokenizer_name_or_path is None:
            raise ValueError("Alpamayo2Super requires vlm_name_or_path or checkpoint path.")
        self.tokenizer = build_alpamayo2_super_tokenizer(
            tokenizer_name_or_path,
            config.history_vocab_size,
            config.future_vocab_size,
        )
        self.history_traj_tokenizer = hyu.instantiate(
            config.hist_traj_tokenizer_cfg,
            load_weights=False,
        )
        self.future_traj_tokenizer = hyu.instantiate(
            config.future_traj_tokenizer_cfg,
            load_weights=False,
        )
        if self.config.enable_expert:
            self.expert = ExpertModel._from_config(self.config.expert_config)
            if not self.config.cotrain_expert_vlm:
                self.vlm.requires_grad_(False)

    def get_output_embeddings(self) -> torch.nn.Module:
        """Return the nested VLM output embeddings."""
        return self.vlm.get_output_embeddings()

    def get_input_embeddings(self) -> torch.nn.Module:
        """Return the nested VLM input embeddings."""
        return self.vlm.get_input_embeddings()

    def tie_weights(self) -> None:
        """Delegate weight tying to the nested VLM model."""
        if hasattr(self, "vlm") and hasattr(self.vlm, "tie_weights"):
            self.vlm.tie_weights()

    def _generate_with_shared_prefill(
        self,
        tokenized_data: dict[str, Any],
        generation_config: GenerationConfig,
        n_samples_total: int,
        **generate_kwargs: Any,
    ) -> GenerateDecoderOnlyOutput:
        """Generate sampled sequences from one shared prompt prefill."""
        generation_config.num_return_sequences = 1
        generation_config.output_logits = False

        input_ids = tokenized_data["input_ids"]
        prompt_attention_mask = tokenized_data["attention_mask"]
        vision_keys = ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")
        vision_inputs = {key: tokenized_data[key] for key in vision_keys if key in tokenized_data}
        prefill_outputs = self.vlm.model(
            input_ids=input_ids[:, :-1],
            attention_mask=prompt_attention_mask[:, :-1],
            use_cache=True,
            **vision_inputs,
        )
        prompt_cache = prefill_outputs.past_key_values
        del prefill_outputs
        prompt_cache.batch_repeat_interleave(n_samples_total)

        vlm_outputs = self.vlm.generate(
            input_ids=torch.repeat_interleave(input_ids, n_samples_total, dim=0),
            attention_mask=torch.repeat_interleave(
                prompt_attention_mask,
                n_samples_total,
                dim=0,
            ),
            past_key_values=prompt_cache,
            generation_config=generation_config,
            **generate_kwargs,
        )
        vlm_outputs.rope_deltas = torch.repeat_interleave(
            self.vlm.model.rope_deltas,
            n_samples_total,
            dim=0,
        )
        return vlm_outputs

    def forward(
        self,
        tokenized_data: dict[str, Any],
        traj_data: dict[str, Any],
        labels_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Alpamayo2SuperModelOutput:
        """Run a training-style forward pass."""
        tokenized_data = dict(tokenized_data)
        tokenized_data["input_ids"] = fuse_traj_tokens(
            self.history_traj_tokenizer,
            self.future_traj_tokenizer,
            tokenized_data["input_ids"],
            traj_data,
            self.config.traj_ids,
        )

        labels = tokenized_data["input_ids"].clone()
        if labels_mask is not None:
            labels = torch.where(labels_mask, labels, IGNORE_INDEX)

        outputs = self.vlm(**tokenized_data, use_cache=False, **kwargs)
        logits = outputs.logits
        cap = float(getattr(self.config, "logit_cap", 0.0))
        if cap > 0.0:
            logits = cap * torch.tanh(logits / cap)

        traj_mask = (
            (
                (labels >= self.config.traj_ids["future_id0"])
                & (labels < self.config.traj_ids["future_id0"] + self.config.future_vocab_size)
            )
            | (labels == self.config.traj_ids["future_start"])
            | (labels == self.config.traj_ids["future_end"])
        )
        future_traj_loss = compute_next_token_loss(
            logits,
            labels,
            traj_mask,
        ) * self.config.loss_weights.get("future_traj", 1.0)
        labels[traj_mask] = IGNORE_INDEX
        other_loss = compute_next_token_loss(
            logits,
            labels,
            labels != IGNORE_INDEX,
        ) * self.config.loss_weights.get("others", 1.0)
        return Alpamayo2SuperModelOutput(loss=future_traj_loss + other_loss, logits=logits)

    @torch.inference_mode()
    def sample_trajectories_from_data(
        self,
        data: dict[str, Any],
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 1,
        num_traj_sets: int = 1,
        max_generation_length: int | None = None,
        diffusion_kwargs: dict[str, Any] | None = None,
        return_extra: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, np.ndarray]]
    ):
        """Generate future trajectories with CoT + expert diffusion inference."""
        if not self.config.enable_expert:
            raise ValueError(
                "Alpamayo2Super public inference requires an expert-enabled checkpoint."
            )

        tokenized_data = dict(data["tokenized_data"])
        traj_data = {
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        }
        input_ids = fuse_traj_tokens(
            self.history_traj_tokenizer,
            self.future_traj_tokenizer,
            tokenized_data["input_ids"],
            traj_data,
            self.config.traj_ids,
        )
        tokenized_data["input_ids"] = input_ids

        n_samples_total = num_traj_samples * num_traj_sets
        generation_config = copy.deepcopy(self.vlm.generation_config)
        generation_config.top_p = top_p
        generation_config.temperature = temperature
        generation_config.do_sample = True
        generation_config.max_new_tokens = max_generation_length or max(
            256, self.config.tokens_per_future_traj
        )
        generation_config.output_logits = False
        generation_config.return_dict_in_generate = True
        generation_config.top_k = top_k
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        logits_processor = LogitsProcessorList(
            [
                MaskDiscreteTrajectoryLogitsProcessor(
                    traj_token_offset=min(
                        self.config.traj_ids["history_id0"],
                        self.config.traj_ids["future_id0"],
                    ),
                    traj_vocab_size=self.config.traj_vocab_size,
                )
            ]
        )
        eos_token_id = self.config.traj_ids["future_start"]
        _append_text_eos_mask(
            logits_processor,
            generation_config.eos_token_id,
            preserved_token_id=eos_token_id,
        )
        stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
        vlm_outputs = self._generate_with_shared_prefill(
            tokenized_data,
            generation_config=generation_config,
            n_samples_total=n_samples_total,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
        )

        vlm_outputs.sequences = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        prompt_cache = vlm_outputs.past_key_values
        prefill_seq_len = prompt_cache.get_seq_length()

        device = input_ids.device
        b_star = vlm_outputs.sequences.shape[0]
        n_diffusion_tokens = self.expert.action_space.get_action_space_dims()[0]
        offset = find_eos_offset(
            sequences=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            device=device,
        )
        prefix_mask = tokenized_data.get("attention_mask")
        if prefix_mask is not None:
            prefix_mask = torch.repeat_interleave(prefix_mask, n_samples_total, dim=0)
        position_ids, attention_mask = build_expert_pos_ids_and_attn_mask(
            offset=offset,
            rope_deltas=vlm_outputs.rope_deltas,
            kv_cache_seq_len=prefill_seq_len,
            n_diffusion_tokens=n_diffusion_tokens,
            b_star=b_star,
            device=device,
            prefix_mask=prefix_mask,
        )

        forward_kwargs = {}
        if self.expert.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """Run one expert denoising step."""
            future_token_embeds = self.expert.action_in_proj(x, t)
            if future_token_embeds.dim() == 2:
                future_token_embeds = future_token_embeds.view(b_star, n_diffusion_tokens, -1)

            expert_outputs = self.expert.expert(
                inputs_embeds=future_token_embeds,
                position_ids=position_ids,
                past_key_values=prompt_cache,
                attention_mask=attention_mask,
                use_cache=True,
                **forward_kwargs,
            )
            prompt_cache.crop(prefill_seq_len)
            pred = self.expert.action_out_proj(expert_outputs.last_hidden_state)
            return pred.view(-1, *self.expert.action_space.get_action_space_dims())

        diffusion_kwargs = diffusion_kwargs or {}
        sampled_action = self.expert.diffusion.sample(
            batch_size=b_star,
            step_fn=step_fn,
            device=device,
            return_all_steps=False,
            **diffusion_kwargs,
        )

        hist_xyz_rep = einops.repeat(
            traj_data["ego_history_xyz"][:, -1],
            "b ... -> (b n) ...",
            n=n_samples_total,
        )
        hist_rot_rep = einops.repeat(
            traj_data["ego_history_rot"][:, -1],
            "b ... -> (b n) ...",
            n=n_samples_total,
        )
        pred_xyz, pred_rot = self.expert.action_space.action_to_traj(
            sampled_action,
            hist_xyz_rep,
            hist_rot_rep,
        )

        pred_xyz = einops.rearrange(
            pred_xyz,
            "(b ns nj) ... -> b ns nj ...",
            ns=num_traj_sets,
            nj=num_traj_samples,
        )
        pred_rot = einops.rearrange(
            pred_rot,
            "(b ns nj) ... -> b ns nj ...",
            ns=num_traj_sets,
            nj=num_traj_samples,
        )
        logprob = torch.zeros_like(pred_xyz[..., 0])

        if not return_extra:
            return pred_xyz, pred_rot, logprob

        extra = extract_text_tokens(self.tokenizer, vlm_outputs.sequences)
        for key in extra:
            extra[key] = np.array(extra[key]).reshape(
                [input_ids.shape[0], num_traj_sets, num_traj_samples]
            )
        return pred_xyz, pred_rot, logprob, extra


AutoModel.register(Alpamayo2SuperConfig, Alpamayo2Super)
