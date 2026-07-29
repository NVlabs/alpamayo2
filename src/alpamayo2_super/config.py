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

"""Configuration for the Alpamayo 2 Super release model."""

from collections.abc import Mapping
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig

from alpamayo2_super.models.expert import ExpertModelConfig
from alpamayo2_super.models.utils import SPECIAL_TOKENS


_HISTORY_TRAJECTORY_WAYPOINTS = 16


def build_alpamayo2_super_tokenizer(
    name_or_path: str,
    history_vocab_size: int,
    future_vocab_size: int,
) -> AutoTokenizer:
    """Build the tokenizer used by the released expert model."""
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, fix_mistral_regex=True)
    tokenizer.add_tokens([f"<i{i}>" for i in range(history_vocab_size)])
    tokenizer.add_tokens([f"<i{i + history_vocab_size}>" for i in range(future_vocab_size)])
    tokenizer.add_tokens(list(SPECIAL_TOKENS.values()), special_tokens=True)
    return tokenizer


def resolve_checkpoint_name_or_path(config: PretrainedConfig) -> str | None:
    """Return the checkpoint path/HF id when available, otherwise the base VLM path."""
    name_or_path = getattr(config, "_name_or_path", None)
    if name_or_path:
        return name_or_path
    return getattr(config, "vlm_name_or_path", None)


class Alpamayo2SuperConfig(PretrainedConfig):
    """Configuration for Alpamayo 2 Super release checkpoints."""

    model_type = "alpamayo2_super"
    sub_configs = {"vlm_config": AutoConfig, "expert_config": ExpertModelConfig}

    def __init__(
        self,
        hist_traj_tokenizer_cfg: Mapping[str, Any] | None = None,
        future_traj_tokenizer_cfg: Mapping[str, Any] | None = None,
        vlm_name_or_path: str | None = None,
        history_vocab_size: int = 1000,
        future_vocab_size: int = 3000,
        dtype: str | torch.dtype = "bfloat16",
        padding_side: str = "left",
        min_pixels: int | None = 163840,
        max_pixels: int | None = 196608,
        include_camera_ids: bool = True,
        token_layout: str = "camera_ts",
        frame_label: str = "frame_num",
        tokens_per_history_traj: int = 48,
        tokens_per_future_traj: int = 128,
        loss_weights: dict[str, float] | None = None,
        expert_config: dict[str, Any] | ExpertModelConfig | None = None,
        enable_expert: bool | None = None,
        cotrain_expert_vlm: bool = False,
        **kwargs: Any,
    ) -> None:
        legacy_use_dt = kwargs.pop("use_dt_tokens", None)
        legacy_include_frame_nums = kwargs.pop("include_frame_nums", None)
        super().__init__(dtype=dtype, **kwargs)

        self.vlm_name_or_path = vlm_name_or_path
        self.hist_traj_tokenizer_cfg = hist_traj_tokenizer_cfg or {}
        self.future_traj_tokenizer_cfg = future_traj_tokenizer_cfg or {}
        self.history_vocab_size = history_vocab_size
        self.future_vocab_size = future_vocab_size
        self.traj_vocab_size = history_vocab_size + future_vocab_size
        self.padding_side = padding_side
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.include_camera_ids = include_camera_ids
        self.token_layout = token_layout

        if legacy_include_frame_nums is not None:
            frame_label = "frame_num" if legacy_include_frame_nums else "none"
        if legacy_use_dt is True:
            frame_label = "dt_token"
        if frame_label not in ("none", "frame_num", "dt_token"):
            raise ValueError(
                f"frame_label must be 'none', 'frame_num', or 'dt_token', got {frame_label!r}"
            )
        self.frame_label = frame_label
        self.include_frame_nums = frame_label == "frame_num"

        if token_layout not in ("camera_ts", "timetick_camera", "timetick_ts"):
            raise ValueError(
                "token_layout must be 'camera_ts', 'timetick_camera', or 'timetick_ts', "
                f"got {token_layout!r}"
            )

        predict_yaw = self.hist_traj_tokenizer_cfg.get("predict_yaw", False)
        pad_origin_at_beginning = self.hist_traj_tokenizer_cfg.get("pad_origin_at_beginning", True)
        trajectory_axes = 4 if predict_yaw else 3
        encoded_intervals = (
            _HISTORY_TRAJECTORY_WAYPOINTS
            if pad_origin_at_beginning
            else _HISTORY_TRAJECTORY_WAYPOINTS - 1
        )
        expected_history_tokens = trajectory_axes * encoded_intervals
        if tokens_per_history_traj != expected_history_tokens:
            raise ValueError(
                "tokens_per_history_traj does not match hist_traj_tokenizer_cfg: "
                f"configured={tokens_per_history_traj}, expected={expected_history_tokens}, "
                f"history_waypoints={_HISTORY_TRAJECTORY_WAYPOINTS}, "
                f"predict_yaw={predict_yaw}, "
                f"pad_origin_at_beginning={pad_origin_at_beginning}"
            )
        self.tokens_per_history_traj = tokens_per_history_traj
        self.tokens_per_future_traj = tokens_per_future_traj
        self.loss_weights = loss_weights or {"future_traj": 1.0, "others": 1.0}
        self.cotrain_expert_vlm = cotrain_expert_vlm

        if not hasattr(self, "vlm_config") and vlm_name_or_path is None:
            return

        if not hasattr(self, "vlm_config"):
            tokenizer = build_alpamayo2_super_tokenizer(
                vlm_name_or_path,
                self.history_vocab_size,
                self.future_vocab_size,
            )
            self.traj_ids = self._build_traj_ids(tokenizer)
            self.vlm_config = AutoConfig.from_pretrained(vlm_name_or_path)
            self.vlm_config.text_config.vocab_size = (len(tokenizer) + 128 - 1) // 128 * 128
            self.vlm_class = self.vlm_config.architectures[0]
        elif isinstance(self.vlm_config, dict):
            self.vlm_config = AutoConfig.for_model(**self.vlm_config)

        self.enable_expert = bool(enable_expert)
        if isinstance(expert_config, ExpertModelConfig):
            self.expert_config = expert_config
            self.enable_expert = True
        elif isinstance(expert_config, dict) and expert_config.get("action_space_cfg") is not None:
            self.expert_config = self.sub_configs["expert_config"](**expert_config)
            self.enable_expert = True
        else:
            self.expert_config = self.sub_configs["expert_config"]()

        if not hasattr(self, "vlm_class"):
            self.vlm_class = self.vlm_config.architectures[0]
        if not hasattr(self, "traj_ids") and vlm_name_or_path is not None:
            tokenizer = build_alpamayo2_super_tokenizer(
                vlm_name_or_path,
                self.history_vocab_size,
                self.future_vocab_size,
            )
            self.traj_ids = self._build_traj_ids(tokenizer)

        if hasattr(self, "vlm_config"):
            self.set_dtype(self.dtype)

    def _build_traj_ids(self, tokenizer: AutoTokenizer) -> dict[str, int]:
        """Build trajectory special-token ids from an Alpamayo tokenizer."""
        return {
            "history_pad": tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["traj_history"]),
            "future_pad": tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["traj_future"]),
            "history_start": tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["traj_history_start"]),
            "future_start": tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["traj_future_start"]),
            "history_end": tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["traj_history_end"]),
            "future_end": tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["traj_future_end"]),
            "history_id0": tokenizer.convert_tokens_to_ids("<i0>"),
            "future_id0": tokenizer.convert_tokens_to_ids(f"<i{self.history_vocab_size}>"),
        }

    def set_dtype(self, dtype: str | torch.dtype | None = None) -> None:
        """Set dtype consistently on the wrapper and nested VLM config."""
        if dtype is None:
            dtype = self.dtype
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype
        self.vlm_config.dtype = dtype
        if hasattr(self.vlm_config, "text_config"):
            self.vlm_config.text_config.dtype = dtype
        if hasattr(self.vlm_config, "vision_config"):
            self.vlm_config.vision_config.dtype = dtype
        if getattr(self, "enable_expert", False):
            self.expert_config.llm_config.dtype = dtype


AutoConfig.register("alpamayo2_super", Alpamayo2SuperConfig)
