# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-native camera and temporal profiles for public inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from alpamayo2_super.common.constants import CAMERA_NAMES_TO_INDICES


CANONICAL_CAMERA_IDS = tuple(CAMERA_NAMES_TO_INDICES.values())
CANONICAL_CAMERA_NAMES = tuple(CAMERA_NAMES_TO_INDICES)
CAMERA_NAMES_BY_ID = {
    camera_id: camera_name for camera_name, camera_id in CAMERA_NAMES_TO_INDICES.items()
}


@dataclass(frozen=True)
class InputProfile:
    """Camera IDs and source-frame indices consumed by one inference task."""

    camera_ids: tuple[int, ...]
    frame_indices: tuple[int, ...]

    @property
    def camera_names(self) -> tuple[str, ...]:
        """Return canonical camera names in this profile's input order."""
        return tuple(CAMERA_NAMES_BY_ID[camera_id] for camera_id in self.camera_ids)


VQA_SIX_CAMERA_FOUR_FRAME = InputProfile(
    camera_ids=(0, 1, 2, 3, 4, 5),
    frame_indices=(0, 1, 2, 3),
)
DRIVING_SIX_CAMERA_FOUR_FRAME = InputProfile(
    camera_ids=(0, 1, 2, 3, 5, 6),
    frame_indices=(0, 1, 2, 3),
)

TASK_INPUT_PROFILES: dict[str, InputProfile] = {
    "trajectory": DRIVING_SIX_CAMERA_FOUR_FRAME,
    "meta_action": DRIVING_SIX_CAMERA_FOUR_FRAME,
    "auto_labeling": DRIVING_SIX_CAMERA_FOUR_FRAME,
    "vqa": VQA_SIX_CAMERA_FOUR_FRAME,
    "grounding": DRIVING_SIX_CAMERA_FOUR_FRAME,
}


def _profile_record(profile: InputProfile) -> dict[str, list[int] | list[str]]:
    """Return serializable selected-camera provenance for one task profile."""
    return {
        "camera_ids": list(profile.camera_ids),
        "camera_names": list(profile.camera_names),
        "frame_indices": list(profile.frame_indices),
    }


def input_profile_record(profile: InputProfile) -> dict[str, list[int] | list[str] | int]:
    """Return complete source and selected-camera provenance for one task input."""
    return {
        "source_camera_ids": list(CANONICAL_CAMERA_IDS),
        "source_camera_names": list(CANONICAL_CAMERA_NAMES),
        **_profile_record(profile),
        "num_cameras": len(profile.camera_ids),
        "num_frames_per_camera": len(profile.frame_indices),
    }


def task_profile_provenance() -> dict[str, Any]:
    """Return the canonical source ring and every ordered public task profile."""
    return {
        "canonical_source": {
            "camera_ids": list(CANONICAL_CAMERA_IDS),
            "camera_names": list(CANONICAL_CAMERA_NAMES),
        },
        "capabilities": {
            task: _profile_record(profile) for task, profile in TASK_INPUT_PROFILES.items()
        },
    }


def validate_source_camera_ring(data: dict[str, Any]) -> tuple[list[int], list[str]]:
    """Require the exact ordered seven-camera source ring before task selection."""
    camera_indices = data.get("camera_indices")
    camera_names = data.get("camera_names")
    if not isinstance(camera_indices, torch.Tensor) or camera_indices.ndim != 1:
        raise ValueError("camera_indices must be a one-dimensional tensor")
    source_camera_ids = [int(camera_id) for camera_id in camera_indices.tolist()]
    if not isinstance(camera_names, (list, tuple)) or not all(
        isinstance(camera_name, str) for camera_name in camera_names
    ):
        raise ValueError("source data must include ordered camera_names")
    source_camera_names = list(camera_names)
    if (
        tuple(source_camera_ids) != CANONICAL_CAMERA_IDS
        or tuple(source_camera_names) != CANONICAL_CAMERA_NAMES
    ):
        raise ValueError(
            "source data must contain the canonical camera ring in order: "
            f"ids={list(CANONICAL_CAMERA_IDS)}, names={list(CANONICAL_CAMERA_NAMES)}"
        )
    return source_camera_ids, source_camera_names


def _normalized_frame_indices(frame_indices: tuple[int, ...], frame_count: int) -> list[int]:
    """Resolve negative source-frame indices and reject out-of-range values."""
    normalized = [index if index >= 0 else frame_count + index for index in frame_indices]
    if any(index < 0 or index >= frame_count for index in normalized):
        raise ValueError(
            f"frame indices {frame_indices} are invalid for a source with {frame_count} frames"
        )
    return normalized


def select_input_profile(data: dict[str, Any], profile: InputProfile) -> dict[str, Any]:
    """Select one profile's cameras and frames while preserving timing invariants."""
    source_camera_ids, _ = validate_source_camera_ring(data)
    camera_indices = data["camera_indices"]
    image_frames = data.get("image_frames")
    if not isinstance(camera_indices, torch.Tensor) or camera_indices.ndim != 1:
        raise ValueError("camera_indices must be a one-dimensional tensor")
    if not isinstance(image_frames, torch.Tensor) or image_frames.ndim != 5:
        raise ValueError("image_frames must have shape [camera, frame, channel, height, width]")
    if image_frames.shape[0] != camera_indices.numel():
        raise ValueError("image_frames and camera_indices must contain the same camera count")

    frame_indices = _normalized_frame_indices(profile.frame_indices, image_frames.shape[1])
    camera_positions = [source_camera_ids.index(camera_id) for camera_id in profile.camera_ids]
    camera_selector = torch.tensor(camera_positions, dtype=torch.long, device=camera_indices.device)
    frame_selector = torch.tensor(frame_indices, dtype=torch.long, device=image_frames.device)

    selected = dict(data)
    selected["camera_indices"] = torch.tensor(
        profile.camera_ids,
        dtype=camera_indices.dtype,
        device=camera_indices.device,
    )
    selected["camera_names"] = list(profile.camera_names)
    for key in ("image_frames", "relative_timestamps", "absolute_timestamps"):
        value = data.get(key)
        if not isinstance(value, torch.Tensor) or value.shape[:2] != image_frames.shape[:2]:
            raise ValueError(f"{key} must begin with the source camera and frame dimensions")
        selected[key] = value.index_select(0, camera_selector.to(value.device)).index_select(
            1, frame_selector.to(value.device)
        )

    absolute_timestamps = selected["absolute_timestamps"]
    camera_tmin = int(absolute_timestamps.min().item())
    selected["camera_tmin"] = camera_tmin
    selected["relative_timestamps"] = (absolute_timestamps - camera_tmin).float() * 1e-6

    ego_t0 = data.get("ego_t0")
    if not isinstance(ego_t0, torch.Tensor) or ego_t0.numel() != 1:
        raise ValueError("ego_t0 must contain one timestamp")
    selected["ego_t0_relative"] = (ego_t0.float() - float(camera_tmin)) * 1e-6
    ego_t0_frame_idx = data.get("ego_t0_frame_idx")
    if not isinstance(ego_t0_frame_idx, torch.Tensor):
        raise ValueError("ego_t0_frame_idx must be a tensor")
    selected["ego_t0_frame_idx"] = torch.tensor(
        [len(frame_indices) - 1],
        dtype=ego_t0_frame_idx.dtype,
        device=ego_t0_frame_idx.device,
    )

    calibrations = data.get("camera_calibrations")
    if isinstance(calibrations, dict):
        selected["camera_calibrations"] = {
            camera_id: calibrations[camera_id]
            for camera_id in profile.camera_ids
            if camera_id in calibrations
        }
    selected["input_profile"] = input_profile_record(profile)
    return selected


def assert_task_input(data: dict[str, Any], task: str) -> None:
    """Require one selected task payload to match its ordered camera contract."""
    try:
        profile = TASK_INPUT_PROFILES[task]
    except KeyError as exc:
        raise ValueError(f"task must be one of {tuple(TASK_INPUT_PROFILES)}, got {task!r}") from exc
    camera_indices = data.get("camera_indices")
    camera_names = data.get("camera_names")
    if (
        not isinstance(camera_indices, torch.Tensor)
        or tuple(int(camera_id) for camera_id in camera_indices.tolist()) != profile.camera_ids
        or tuple(camera_names or ()) != profile.camera_names
        or data.get("input_profile") != input_profile_record(profile)
    ):
        raise ValueError(f"task input for {task!r} does not match its ordered camera contract")


def select_task_input(data: dict[str, Any], task: str) -> dict[str, Any]:
    """Select the fixed release input profile for ``task``."""
    try:
        profile = TASK_INPUT_PROFILES[task]
    except KeyError as exc:
        raise ValueError(f"task must be one of {tuple(TASK_INPUT_PROFILES)}, got {task!r}") from exc
    selected = select_input_profile(data, profile)
    assert_task_input(selected, task)
    return selected
