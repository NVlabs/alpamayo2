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

"""Tests for the Alpamayo 2 Super model configuration."""

import pytest

from alpamayo2_super.config import Alpamayo2SuperConfig


@pytest.mark.parametrize(
    ("predict_yaw", "pad_origin_at_beginning", "tokens_per_history_traj"),
    [
        pytest.param(False, False, 45, id="xyz-15-intervals"),
        pytest.param(False, True, 48, id="xyz-16-intervals"),
        pytest.param(True, False, 60, id="xyzyaw-15-intervals"),
        pytest.param(True, True, 64, id="xyzyaw-16-intervals"),
    ],
)
def test_history_token_count_accepts_tokenizer_option_pairs(
    predict_yaw: bool,
    pad_origin_at_beginning: bool,
    tokens_per_history_traj: int,
) -> None:
    """All tokenizer option pairs accept their 16-waypoint encoded token count."""
    config = Alpamayo2SuperConfig(
        hist_traj_tokenizer_cfg={
            "predict_yaw": predict_yaw,
            "pad_origin_at_beginning": pad_origin_at_beginning,
        },
        tokens_per_history_traj=tokens_per_history_traj,
    )

    assert config.tokens_per_history_traj == tokens_per_history_traj


@pytest.mark.parametrize(
    ("hist_traj_tokenizer_cfg", "configured", "expected", "pad_origin_at_beginning"),
    [
        pytest.param(
            {"predict_yaw": False, "pad_origin_at_beginning": False},
            48,
            45,
            False,
            id="configured-48-expected-45",
        ),
        pytest.param(
            {},
            45,
            48,
            True,
            id="configured-45-expected-48-default-options",
        ),
    ],
)
def test_history_token_count_rejects_mismatch_without_vlm_config(
    hist_traj_tokenizer_cfg: dict[str, bool],
    configured: int,
    expected: int,
    pad_origin_at_beginning: bool,
) -> None:
    """Construction fails before the no-VLM early return with actionable details."""
    with pytest.raises(ValueError) as exc_info:
        Alpamayo2SuperConfig(
            hist_traj_tokenizer_cfg=hist_traj_tokenizer_cfg,
            tokens_per_history_traj=configured,
        )

    message = str(exc_info.value)
    assert f"configured={configured}" in message
    assert f"expected={expected}" in message
    assert "predict_yaw=False" in message
    assert f"pad_origin_at_beginning={pad_origin_at_beginning}" in message
