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

"""Tests for release-native delta trajectory tokenization."""

import pytest
import torch

from alpamayo2_super.models.delta_tokenizer import DeltaTrajectoryTokenizer


def _constant_speed_trajectory(start_x: float, end_x: float, num_points: int) -> torch.Tensor:
    """Return a straight-line trajectory along +x with shape ``(1, T, 3)``."""
    xyz = torch.zeros(1, num_points, 3)
    xyz[0, :, 0] = torch.linspace(start_x, end_x, num_points)
    return xyz


def _identity_rot(num_points: int) -> torch.Tensor:
    """Return identity rotations for one trajectory with ``num_points`` poses."""
    return torch.eye(3).expand(1, num_points, 3, 3).contiguous()


def _tokens_per_waypoint_axis(tokenizer: DeltaTrajectoryTokenizer, num_points: int) -> int:
    """Return deltas per axis under the configured anchoring convention."""
    return num_points if tokenizer.pad_origin_at_beginning else num_points - 1


def _roundtrip(tokenizer: DeltaTrajectoryTokenizer, xyz: torch.Tensor) -> torch.Tensor:
    """Encode and decode ``xyz`` while checking the expected tensor shapes."""
    num_points = xyz.shape[1]
    rot = _identity_rot(num_points)
    tokens = tokenizer.encode(xyz, rot, xyz, rot)
    assert tokens.shape == (1, _tokens_per_waypoint_axis(tokenizer, num_points) * 3)
    decoded_xyz, _, _ = tokenizer.decode(xyz, rot, tokens)
    assert decoded_xyz.shape == xyz.shape
    return decoded_xyz


_ROUNDTRIP_TOLERANCE_M = 0.25


def test_roundtrip_future_pad_origin_at_beginning() -> None:
    """Legacy mode reconstructs a future trajectory that starts at the ego origin."""
    future_xyz = _constant_speed_trajectory(start_x=1.4, end_x=22.4, num_points=16)
    decoded_xyz = _roundtrip(DeltaTrajectoryTokenizer(), future_xyz)
    assert (decoded_xyz - future_xyz).abs().max() < _ROUNDTRIP_TOLERANCE_M


def test_roundtrip_history_anchored_at_terminal_origin() -> None:
    """History mode reconstructs T points from T-1 deltas and an exact terminal origin."""
    history_xyz = _constant_speed_trajectory(start_x=-21.0, end_x=0.0, num_points=16)
    tokenizer = DeltaTrajectoryTokenizer(pad_origin_at_beginning=False)
    decoded_xyz = _roundtrip(tokenizer, history_xyz)
    assert (decoded_xyz - history_xyz).abs().max() < _ROUNDTRIP_TOLERANCE_M
    assert decoded_xyz[0, -1].abs().max() == 0.0


def test_terminal_anchoring_avoids_saturating_first_history_delta() -> None:
    """History mode keeps local deltas in range instead of clipping the oldest point."""
    history_xyz = _constant_speed_trajectory(start_x=-21.0, end_x=0.0, num_points=16)
    rot = _identity_rot(16)

    legacy_tokenizer = DeltaTrajectoryTokenizer()
    legacy_tokens = legacy_tokenizer.encode(history_xyz, rot, history_xyz, rot)
    assert legacy_tokens[0, 0] == 0
    legacy_xyz, _, _ = legacy_tokenizer.decode(history_xyz, rot, legacy_tokens)
    assert (legacy_xyz - history_xyz).abs().max() > 10.0

    anchored_tokens = DeltaTrajectoryTokenizer(pad_origin_at_beginning=False).encode(
        history_xyz, rot, history_xyz, rot
    )
    assert anchored_tokens.shape == (1, 15 * 3)
    assert (anchored_tokens == 0).sum() == 0
    assert (anchored_tokens == 999).sum() == 0


def _arc_trajectory_with_yaw(
    num_points: int, end_at_origin: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a gentle left arc and rotations in the tokenizer's yaw convention."""
    yaw_start = -0.6 if end_at_origin else 0.0
    yaw_end = 0.0 if end_at_origin else 0.6
    yaw = torch.linspace(yaw_start, yaw_end, num_points).unsqueeze(0)
    step_xy = 1.2 * torch.stack([torch.cos(yaw[0]), torch.sin(yaw[0])], dim=-1)
    xyz = torch.zeros(1, num_points, 3)
    xyz[0, :, :2] = torch.cumsum(step_xy, dim=0)
    if end_at_origin:
        xyz = xyz - xyz[:, -1:, :]

    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
    rot = torch.zeros(1, num_points, 3, 3)
    rot[..., 0, 0] = cos_yaw
    rot[..., 0, 1] = sin_yaw
    rot[..., 1, 0] = -sin_yaw
    rot[..., 1, 1] = cos_yaw
    rot[..., 2, 2] = 1.0
    return xyz, yaw, rot


@pytest.mark.parametrize("pad_origin_at_beginning", [True, False])
def test_roundtrip_with_yaw(pad_origin_at_beginning: bool) -> None:
    """Yaw and positions round-trip under both anchoring conventions."""
    xyz, yaw, rot = _arc_trajectory_with_yaw(16, end_at_origin=not pad_origin_at_beginning)
    tokenizer = DeltaTrajectoryTokenizer(
        predict_yaw=True,
        pad_origin_at_beginning=pad_origin_at_beginning,
    )
    tokens = tokenizer.encode(xyz, rot, xyz, rot)
    assert tokens.shape == (1, _tokens_per_waypoint_axis(tokenizer, 16) * 4)

    decoded_xyz, decoded_rot, _ = tokenizer.decode(xyz, rot, tokens)
    assert (decoded_xyz - xyz).abs().max() < _ROUNDTRIP_TOLERANCE_M
    decoded_yaw = torch.atan2(decoded_rot[..., 1, 0], decoded_rot[..., 0, 0])
    assert (decoded_yaw - yaw).abs().max() < 0.1
