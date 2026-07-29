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

"""Utilities for expert trajectory generation."""

import logging

import einops
import torch
from transformers import StoppingCriteria

logger = logging.getLogger(__name__)


class StopAfterEOS(StoppingCriteria):
    """Stop generation after every sequence has emitted ``eos_token_id``."""

    def __init__(self, eos_token_id: int) -> None:
        self.eos_token_id = eos_token_id
        self.eos_found: torch.Tensor | None = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """Update EOS state and report whether generation can stop."""
        del scores, kwargs
        if self.eos_found is None:
            self.eos_found = torch.zeros(
                input_ids.shape[0],
                dtype=torch.bool,
                device=input_ids.device,
            )

        if self.eos_found.all():
            return True

        self.eos_found = self.eos_found | (input_ids[:, -1] == self.eos_token_id)
        return False


def replace_padding_after_eos(
    token_ids: torch.Tensor,
    eos_token_id: int,
    pad_token_id: int,
) -> torch.Tensor:
    """Overwrite tokens after the first EOS in each sequence with padding."""
    batch_size, seq_len = token_ids.shape
    eos_mask = token_ids == eos_token_id
    eos_positions = torch.where(
        eos_mask,
        torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1),
        torch.full((batch_size, seq_len), seq_len, dtype=torch.long, device=token_ids.device),
    )
    first_eos_pos = eos_positions.min(dim=1, keepdim=True)[0]
    position_indices = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
    token_ids[position_indices > first_eos_pos] = pad_token_id
    return token_ids


def find_eos_offset(
    sequences: torch.Tensor,
    eos_token_id: int,
    device: torch.device,
    warn: bool = True,
) -> torch.Tensor:
    """Return the token offset right after the first EOS in each sequence."""
    mask = sequences == eos_token_id
    has_eos = mask.any(dim=1)
    if warn:
        for idx in range(sequences.shape[0]):
            if not has_eos[idx]:
                logger.warning(
                    "No <traj_future_start> token found in generated sequence %s; "
                    "using the final generated token as the expert boundary.",
                    idx,
                )
    eos_positions = mask.int().argmax(dim=1)
    last_positions = torch.full((sequences.shape[0],), sequences.shape[1] - 1, device=device)
    return torch.where(has_eos, eos_positions, last_positions) + 1


def build_expert_pos_ids_and_attn_mask(
    offset: torch.Tensor,
    rope_deltas: torch.Tensor,
    kv_cache_seq_len: int,
    n_diffusion_tokens: int,
    b_star: int,
    device: torch.device,
    prefix_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Qwen-VL position IDs and cache attention mask for expert denoising."""
    position_ids = torch.arange(n_diffusion_tokens, device=device)
    position_ids = einops.repeat(position_ids, "l -> 3 b l", b=b_star).clone()
    position_ids += (rope_deltas + offset[:, None]).to(position_ids.device)

    attention_mask = torch.zeros(
        (b_star, 1, n_diffusion_tokens, kv_cache_seq_len + n_diffusion_tokens),
        dtype=torch.float32,
        device=device,
    )
    min_value = torch.finfo(attention_mask.dtype).min
    for idx in range(b_star):
        attention_mask[idx, :, :, offset[idx] : -n_diffusion_tokens] = min_value

    if prefix_mask is not None:
        input_mask = prefix_mask[:, None, None, :]
        attention_mask[:, :, :, : input_mask.shape[-1]] = torch.where(
            input_mask == 0,
            min_value,
            attention_mask[:, :, :, : input_mask.shape[-1]],
        )

    return position_ids, attention_mask
