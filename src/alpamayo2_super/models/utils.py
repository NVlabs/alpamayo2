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

"""Shared token utilities for the Alpamayo 2 Super release model."""

from collections.abc import Mapping
from typing import Any

import einops
import torch

IGNORE_INDEX = -100

SPECIAL_TOKENS_KEYS = [
    "prompt_start",
    "prompt_end",
    "image_start",
    "image_pre_tkn",
    "image_end",
    "traj_history_start",
    "traj_history_pre_tkn",
    "traj_history_end",
    "cot_start",
    "cot_end",
    "meta_action_start",
    "meta_action_end",
    "traj_future_start",
    "traj_future_pre_tkn",
    "traj_future_end",
    "traj_history",
    "traj_future",
    "image_pad",
    "vectorized_wm",
    "vectorized_wm_start",
    "vectorized_wm_end",
    "vectorized_wm_pre_tkn",
    "route_start",
    "route_pad",
    "route_end",
    "question_start",
    "question_end",
    "answer_start",
    "answer_end",
]
SPECIAL_TOKENS = {key: f"<|{key}|>" for key in SPECIAL_TOKENS_KEYS}


def replace_pad_token(input_ids: torch.Tensor, new_ids: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """Replace each row's trajectory placeholders with that row's encoded trajectory ids.

    Args:
        input_ids: (B, L) token ids containing ``pad_idx`` placeholders.
        new_ids: (B, K) encoded trajectory ids.
        pad_idx: Placeholder token id to replace.

    Returns:
        (B, L) token ids with trajectory placeholders replaced.

    Raises:
        ValueError: The tensors are not rank two, have different batch sizes, or a row's
            placeholder count differs from its encoded id count.
    """
    if input_ids.ndim != 2 or new_ids.ndim != 2:
        raise ValueError(
            "replace_pad_token requires rank-2 input_ids and new_ids shaped [batch, tokens]; "
            f"got input_ids.shape={tuple(input_ids.shape)}, new_ids.shape={tuple(new_ids.shape)}"
        )
    if input_ids.shape[0] != new_ids.shape[0]:
        raise ValueError(
            "replace_pad_token requires input_ids and new_ids to have the same batch size; "
            f"got input_ids.shape={tuple(input_ids.shape)}, new_ids.shape={tuple(new_ids.shape)}"
        )

    mask = input_ids == pad_idx
    placeholder_counts = mask.sum(dim=1)
    encoded_ids_per_row = new_ids.shape[1]
    mismatched_rows_mask = placeholder_counts != encoded_ids_per_row
    if mismatched_rows_mask.any():
        placeholder_counts_list = placeholder_counts.tolist()
        mismatched_rows = mismatched_rows_mask.nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(
            "replace_pad_token requires exact per-row token counts: "
            f"mismatched rows={mismatched_rows}, "
            f"placeholder counts={placeholder_counts_list}, "
            f"new_ids tokens per row={encoded_ids_per_row}, pad_idx={pad_idx}"
        )

    return input_ids.masked_scatter(mask, new_ids)


def tokenize_history_trajectory(
    tokenizer: Any,
    traj_data: Mapping[str, Any],
    start_idx: int = 0,
) -> torch.Tensor:
    """Tokenize history trajectory tensors with prefix shape ``[B, n_traj, ...]``."""
    if "ego_history_xyz" not in traj_data:
        raise KeyError("traj_data must include ego_history_xyz")
    if traj_data["ego_history_xyz"].ndim != 4:
        raise ValueError("ego_history_xyz must be 4D with shape [B, n_traj, T, 3]")

    batch_size = traj_data["ego_history_xyz"].shape[0]
    hist_xyz = traj_data["ego_history_xyz"].flatten(start_dim=0, end_dim=1)
    hist_rot = traj_data["ego_history_rot"].flatten(start_dim=0, end_dim=1)

    hist_idx = (
        tokenizer.encode(
            hist_xyz=hist_xyz[:, :1],
            hist_rot=hist_rot[:, :1],
            fut_xyz=hist_xyz,
            fut_rot=hist_rot,
        )
        + start_idx
    )
    return einops.rearrange(hist_idx, "(b n_traj) n -> b (n_traj n)", b=batch_size)


def tokenize_future_trajectory(
    tokenizer: Any,
    traj_data: Mapping[str, Any],
    start_idx: int = 0,
) -> torch.Tensor:
    """Tokenize future trajectory tensors with prefix shape ``[B, n_traj, ...]``."""
    if "ego_future_xyz" not in traj_data:
        raise KeyError("traj_data must include ego_future_xyz")
    if traj_data["ego_future_xyz"].ndim != 4:
        raise ValueError("ego_future_xyz must be 4D with shape [B, n_traj, T, 3]")

    batch_size = traj_data["ego_future_xyz"].shape[0]
    hist_xyz = traj_data["ego_history_xyz"].flatten(start_dim=0, end_dim=1)
    hist_rot = traj_data["ego_history_rot"].flatten(start_dim=0, end_dim=1)
    fut_xyz = traj_data["ego_future_xyz"].flatten(start_dim=0, end_dim=1)
    fut_rot = traj_data["ego_future_rot"].flatten(start_dim=0, end_dim=1)

    fut_idx = (
        tokenizer.encode(hist_xyz=hist_xyz, hist_rot=hist_rot, fut_xyz=fut_xyz, fut_rot=fut_rot)
        + start_idx
    )
    return einops.rearrange(fut_idx, "(b n_traj) n -> b (n_traj n)", b=batch_size)


def fuse_traj_tokens(
    history_traj_tokenizer: Any,
    future_traj_tokenizer: Any,
    input_ids: torch.Tensor,
    traj_data: Mapping[str, Any],
    traj_ids: Mapping[str, int],
) -> torch.Tensor:
    """Fuse encoded trajectory ids into history/future placeholder spans."""
    hist_idx = tokenize_history_trajectory(
        history_traj_tokenizer,
        traj_data,
        traj_ids["history_id0"],
    )
    input_ids = replace_pad_token(input_ids, hist_idx, traj_ids["history_pad"])

    if "ego_future_xyz" in traj_data:
        future_idx = tokenize_future_trajectory(
            future_traj_tokenizer,
            traj_data,
            traj_ids["future_id0"],
        )
        input_ids = replace_pad_token(input_ids, future_idx, traj_ids["future_pad"])

    return input_ids


def compute_next_token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    labels_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute masked next-token cross entropy."""
    if labels_mask is None:
        labels_mask = torch.ones_like(labels, dtype=torch.bool)
    if labels_mask[:, 1:].sum() == 0:
        return torch.tensor(0.0, device=labels.device)

    shift_labels = labels[..., 1:]
    shift_logits = logits[..., :-1, :]
    shift_labels = shift_labels[labels_mask[:, 1:]].contiguous()
    shift_logits = shift_logits[labels_mask[:, 1:]].contiguous().float()
    shift_labels = shift_labels.to(shift_logits.device)
    return torch.nan_to_num(
        torch.nn.functional.cross_entropy(
            shift_logits,
            shift_labels,
            ignore_index=IGNORE_INDEX,
            reduction="mean",
        ),
        nan=0.0,
    )
