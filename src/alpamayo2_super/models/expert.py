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

"""Expert denoiser used by Alpamayo 2 Super trajectory inference."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import hydra.utils as hyu
import torch
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.utils.generic import ModelOutput

from alpamayo2_super.action_space.action_space import ActionSpace
from alpamayo2_super.diffusion.base import BaseDiffusion


class ExpertModelConfig(PretrainedConfig):
    """Configuration for the action expert."""

    model_type = "alpamayo2_super_expert"
    sub_configs = {"llm_config": AutoConfig}

    def __init__(
        self,
        llm_config: AutoConfig | Mapping[str, Any] | None = None,
        action_space_cfg: Mapping[str, Any] | None = None,
        diffusion_cfg: Mapping[str, Any] | None = None,
        action_in_proj_cfg: Mapping[str, Any] | None = None,
        action_out_proj_cfg: Mapping[str, Any] | None = None,
        expert_non_causal_attention: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if isinstance(llm_config, dict):
            llm_config = AutoConfig.for_model(**llm_config)
        self.llm_config = llm_config
        self.action_space_cfg = action_space_cfg
        self.diffusion_cfg = diffusion_cfg
        self.action_in_proj_cfg = action_in_proj_cfg
        self.action_out_proj_cfg = action_out_proj_cfg
        self.expert_non_causal_attention = expert_non_causal_attention


@dataclass
class ExpertModelOutput(ModelOutput):
    """Output from the expert denoiser."""

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None


class ExpertModel(PreTrainedModel):
    """Action expert that denoises future trajectory actions conditioned on VLM cache."""

    config_class = ExpertModelConfig
    base_model_prefix = "expert"
    _supports_flash_attn = True
    _supports_sdpa = True

    def __init__(self, config: ExpertModelConfig, dtype: str = "bfloat16", **kwargs: Any) -> None:
        super().__init__(config=config, dtype=dtype, **kwargs)
        self.expert = AutoModel.from_config(config.llm_config)
        del self.expert.embed_tokens

        self.action_space: ActionSpace = hyu.instantiate(config.action_space_cfg)
        self.diffusion: BaseDiffusion = hyu.instantiate(
            config.diffusion_cfg,
            x_dims=self.action_space.get_action_space_dims(),
        )
        if self.diffusion.use_classifier_free_guidance:
            raise ValueError("Classifier-free guidance is not supported by Alpamayo2Super.")

        self.action_in_proj = hyu.instantiate(
            config.action_in_proj_cfg,
            in_dims=self.action_space.get_action_space_dims(),
            out_dim=self.expert.config.hidden_size,
        )
        self.action_out_proj = hyu.instantiate(
            config.action_out_proj_cfg,
            in_features=self.expert.config.hidden_size,
            out_features=self.action_space.get_action_space_dims()[-1],
        )

    def _process_traj_future_training(self, traj_data: dict[str, Any]) -> dict[str, Any]:
        """Convert ground-truth future trajectories into noisy expert training targets."""
        action = self.action_space.traj_to_action(
            traj_history_xyz=traj_data["ego_history_xyz"],
            traj_history_rot=traj_data["ego_history_rot"],
            traj_future_xyz=traj_data["ego_future_xyz"],
            traj_future_rot=traj_data["ego_future_rot"],
        )
        action = action.reshape(-1, *self.action_space.get_action_space_dims())
        return self.diffusion.construct_training_data(action)

    def _process_mrope_position_ids(
        self,
        vlm_outputs: Any,
        batch_size: int,
        num_expert_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build Qwen-VL MRoPE position IDs for expert tokens."""
        position_ids = torch.arange(num_expert_tokens, device=device)
        position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1).clone()
        delta = vlm_outputs.rope_deltas + vlm_outputs.past_key_values.get_seq_length()
        position_ids += delta.to(position_ids.device)
        return position_ids

    def forward(
        self,
        traj_data: dict[str, Any],
        vlm_outputs: Any,
        cache_attention_mask: torch.Tensor | None = None,
    ) -> ExpertModelOutput:
        """Run the expert training-style forward pass."""
        batch_size = vlm_outputs.logits.shape[0]
        future_traj_data = self._process_traj_future_training(traj_data)
        expert_embeds = self.action_in_proj(
            future_traj_data["noisy_x"],
            future_traj_data["timesteps"],
        )
        position_ids = self._process_mrope_position_ids(
            vlm_outputs,
            batch_size,
            expert_embeds.shape[1],
            expert_embeds.device,
        )
        attention_mask = None
        if cache_attention_mask is not None:
            expert_token_mask = cache_attention_mask.new_ones((batch_size, expert_embeds.shape[1]))
            attention_mask = torch.cat([cache_attention_mask, expert_token_mask], dim=1)

        forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False
        expert_outputs = self.expert(
            inputs_embeds=expert_embeds,
            position_ids=position_ids,
            past_key_values=vlm_outputs.past_key_values,
            attention_mask=attention_mask,
            use_cache=True,
            **forward_kwargs,
        )
        pred = self.action_out_proj(expert_outputs.last_hidden_state)
        pred = pred.view(-1, *self.action_space.get_action_space_dims())
        loss = self.diffusion.compute_loss_from_pred(future_traj_data, pred.float())
        return ExpertModelOutput(loss=loss, logits=pred)


AutoConfig.register("alpamayo2_super_expert", ExpertModelConfig)
AutoModel.register(ExpertModelConfig, ExpertModel)
