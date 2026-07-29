# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for public task-native camera and frame profiles."""

from __future__ import annotations

import pytest
import torch

from alpamayo2_super.common.constants import CAMERA_NAMES_TO_INDICES
from alpamayo2_super.input_profiles import TASK_INPUT_PROFILES, select_task_input


@pytest.fixture
def seven_camera_data() -> dict:
    """Build a timestamped seven-camera/four-frame source payload."""
    camera_count = 7
    frame_count = 4
    image_frames = torch.arange(camera_count * frame_count).reshape(
        camera_count, frame_count, 1, 1, 1
    )
    absolute_timestamps = torch.tensor(
        [[4_800_000, 4_900_000, 5_000_000, 5_100_000]] * camera_count,
        dtype=torch.int64,
    )
    return {
        "image_frames": image_frames,
        "camera_indices": torch.arange(camera_count, dtype=torch.int64),
        "camera_names": list(CAMERA_NAMES_TO_INDICES),
        "absolute_timestamps": absolute_timestamps,
        "relative_timestamps": (absolute_timestamps - 4_800_000).float() * 1e-6,
        "camera_tmin": 4_800_000,
        "ego_t0": torch.tensor([5_100_000], dtype=torch.int64),
        "ego_t0_relative": torch.tensor([0.3]),
        "ego_t0_frame_idx": torch.tensor([3], dtype=torch.int64),
        "camera_calibrations": {camera_id: {"camera_id": camera_id} for camera_id in range(7)},
    }


@pytest.mark.parametrize(
    ("task", "camera_ids"),
    [
        ("trajectory", [0, 1, 2, 3, 5, 6]),
        ("meta_action", [0, 1, 2, 3, 5, 6]),
        ("auto_labeling", [0, 1, 2, 3, 5, 6]),
        ("vqa", [0, 1, 2, 3, 4, 5]),
        ("grounding", [0, 1, 2, 3, 5, 6]),
    ],
)
def test_select_task_input_applies_validated_six_camera_four_frame_profile(
    seven_camera_data: dict,
    task: str,
    camera_ids: list[int],
) -> None:
    """Each public capability receives its validated ordered input profile."""
    selected = select_task_input(seven_camera_data, task)

    assert selected["camera_indices"].tolist() == camera_ids
    assert selected["image_frames"].shape[:2] == (6, 4)
    assert selected["input_profile"]["camera_ids"] == camera_ids
    assert selected["input_profile"]["frame_indices"] == [0, 1, 2, 3]
    for output_position, source_camera_id in enumerate(camera_ids):
        torch.testing.assert_close(
            selected["image_frames"][output_position],
            seven_camera_data["image_frames"][source_camera_id],
        )


def test_select_task_input_preserves_timing_and_filters_calibration(
    seven_camera_data: dict,
) -> None:
    """Selection recomputes timing and retains calibration for consumed cameras only."""
    selected = select_task_input(seven_camera_data, "trajectory")

    assert selected["camera_tmin"] == 4_800_000
    torch.testing.assert_close(
        selected["relative_timestamps"],
        torch.tensor([[0.0, 0.1, 0.2, 0.3]] * 6),
    )
    torch.testing.assert_close(selected["ego_t0_relative"], torch.tensor([0.3]))
    assert selected["ego_t0_frame_idx"].tolist() == [3]
    assert list(selected["camera_calibrations"]) == [0, 1, 2, 3, 5, 6]


def test_select_task_input_does_not_mutate_source(seven_camera_data: dict) -> None:
    """Multiple task selections may safely share one loaded source sample."""
    source_camera_indices = seven_camera_data["camera_indices"].clone()
    source_image_frames = seven_camera_data["image_frames"].clone()

    select_task_input(seven_camera_data, "vqa")
    select_task_input(seven_camera_data, "grounding")

    torch.testing.assert_close(seven_camera_data["camera_indices"], source_camera_indices)
    torch.testing.assert_close(seven_camera_data["image_frames"], source_image_frames)


def test_select_task_input_rejects_unknown_task(seven_camera_data: dict) -> None:
    """An unknown task fails before model preparation."""
    with pytest.raises(ValueError, match="task must be one of"):
        select_task_input(seven_camera_data, "captioning")


def test_public_profiles_cover_exact_release_capabilities() -> None:
    """The public package exposes only the five validated inference capabilities."""
    assert set(TASK_INPUT_PROFILES) == {
        "trajectory",
        "meta_action",
        "auto_labeling",
        "vqa",
        "grounding",
    }
