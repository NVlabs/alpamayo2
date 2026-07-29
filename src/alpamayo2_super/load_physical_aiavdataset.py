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

"""Load data from physical_ai_av.PhysicalAIAVDatasetInterface for model inference."""

import logging
from typing import Any

import numpy as np
import physical_ai_av
import scipy.spatial.transform as spt
import torch
from einops import rearrange

from alpamayo2_super.common.constants import CAMERA_NAMES_TO_INDICES

logger = logging.getLogger(__name__)


def load_physical_aiavdataset(
    clip_id: str,
    t0_us: int = 5_100_000,
    avdi: physical_ai_av.PhysicalAIAVDatasetInterface | None = None,
    maybe_stream: bool = True,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    time_step: float = 0.1,
    camera_features: list | None = None,
    num_frames: int = 4,
    include_calibration: bool = True,
) -> dict[str, Any]:
    """Load data from physical_ai_av for model inference.

    This function loads a sample from the physical_ai_av dataset and converts it
    to the format expected by Alpamayo2Super model inference.

    Args:
        clip_id: The clip ID to load data from. Use the checked-in validation manifests
            or the PhysicalAI-AV clip index.
        t0_us: The clip-relative timestamp (in microseconds) at which to sample
            the trajectory.
        avdi: Optional pre-initialized PhysicalAIAVDatasetInterface. If None, creates one.
        maybe_stream: Whether to stream data from HuggingFace (if not downloaded locally).
        num_history_steps: Number of history trajectory steps (default: 16 for 1.6s at 10Hz).
        num_future_steps: Number of future trajectory steps (default: 64 for 6.4s at 10Hz).
        time_step: Time step between trajectory points in seconds (default: 0.1s = 10Hz).
        camera_features: List of camera features to load. If None, uses the 7-camera
            ring used by the Alpamayo 2 Super S1 configuration.
        num_frames: Number of frames per camera to load (default: 4).
        include_calibration: Best-effort load of camera intrinsics/extrinsics for
            visualization overlays. Missing calibration falls back to BEV-only visualization.

    Returns:
        A dictionary with the following keys:
            - image_frames: torch.Tensor of shape (N_cameras, num_frames, 3, H, W)
            - camera_indices: torch.Tensor of shape (N_cameras,)
            - ego_history_xyz: torch.Tensor of shape (1, 1, num_history_steps, 3)
            - ego_history_rot: torch.Tensor of shape (1, 1, num_history_steps, 3, 3)
            - ego_future_xyz: torch.Tensor of shape (1, 1, num_future_steps, 3)
            - ego_future_rot: torch.Tensor of shape (1, 1, num_future_steps, 3, 3)
            - relative_timestamps: torch.Tensor of shape (N_cameras, num_frames)
            - absolute_timestamps: torch.Tensor of shape (N_cameras, num_frames)
            - camera_tmin: Minimum image timestamp in the clip-relative time base
            - ego_t0: torch.Tensor of shape (1,) in the clip-relative time base
            - ego_t0_relative: torch.Tensor of shape (1,) in seconds from camera_tmin
            - t0_us: The t0 timestamp used
            - clip_id: The clip ID
    """
    if avdi is None:
        avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    if camera_features is None:
        camera_features = [
            avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
            avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
            avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
            avdi.features.CAMERA.CAMERA_REAR_LEFT_70FOV,
            avdi.features.CAMERA.CAMERA_REAR_TELE_30FOV,
            avdi.features.CAMERA.CAMERA_REAR_RIGHT_70FOV,
            avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
        ]

    camera_name_to_index = CAMERA_NAMES_TO_INDICES
    camera_index_to_name = {idx: name for name, idx in camera_name_to_index.items()}
    camera_features_by_name = {}
    for camera_feature in camera_features:
        if not isinstance(camera_feature, str):
            raise ValueError(f"Unexpected camera feature type: {type(camera_feature)}")
        camera_name = camera_feature.rsplit("/", maxsplit=1)[-1].lower()
        if camera_name not in camera_name_to_index:
            raise ValueError(f"unknown camera feature name {camera_name!r}")
        camera_features_by_name[camera_feature] = camera_name

    # Load egomotion data
    egomotion = avdi.get_clip_feature(
        clip_id,
        avdi.features.LABELS.EGOMOTION,
        maybe_stream=maybe_stream,
    )

    assert t0_us > num_history_steps * time_step * 1_000_000, (
        "t0_us must be greater than the history time range"
    )

    # Compute timestamps for trajectory sampling
    # History: [..., t0-0.2s, t0-0.1s, t0] (num_history_steps points ending at t0)
    # Future: [t0+0.1s, t0+0.2s, ..., t0+6.4s] (num_future_steps points after t0)
    history_offsets_us = np.arange(
        -(num_history_steps - 1) * time_step * 1_000_000,
        time_step * 1_000_000 / 2,
        time_step * 1_000_000,
    ).astype(np.int64)
    history_timestamps = t0_us + history_offsets_us

    future_offsets_us = np.arange(
        time_step * 1_000_000,
        (num_future_steps + 0.5) * time_step * 1_000_000,
        time_step * 1_000_000,
    ).astype(np.int64)
    future_timestamps = t0_us + future_offsets_us

    # Get egomotion at history and future timestamps
    ego_history = egomotion(history_timestamps)
    ego_history_xyz = ego_history.pose.translation  # (num_history_steps, 3)
    ego_history_quat = ego_history.pose.rotation.as_quat()  # (num_history_steps, 4)

    ego_future = egomotion(future_timestamps)
    ego_future_xyz = ego_future.pose.translation  # (num_future_steps, 3)
    ego_future_quat = ego_future.pose.rotation.as_quat()  # (num_future_steps, 4)

    # Transform to local frame (relative to t0 pose)
    # The model expects trajectories in the ego frame at t0.
    # Transformation: xyz_local = R_t0^{-1} @ (xyz_world - xyz_t0)
    t0_xyz = ego_history_xyz[-1].copy()  # Position at t0
    t0_quat = ego_history_quat[-1].copy()  # Orientation at t0
    t0_rot = spt.Rotation.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()

    # Transform history positions to local frame
    ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)

    # Transform future positions to local frame
    ego_future_xyz_local = t0_rot_inv.apply(ego_future_xyz - t0_xyz)

    # Transform rotations to local frame
    ego_history_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_history_quat)).as_matrix()
    ego_future_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_future_quat)).as_matrix()

    # Convert to torch tensors with batch dimensions: (B=1, n_traj_group=1, T, ...)
    ego_history_xyz_tensor = (
        torch.from_numpy(ego_history_xyz_local).float().unsqueeze(0).unsqueeze(0)
    )
    ego_history_rot_tensor = (
        torch.from_numpy(ego_history_rot_local).float().unsqueeze(0).unsqueeze(0)
    )
    ego_future_xyz_tensor = torch.from_numpy(ego_future_xyz_local).float().unsqueeze(0).unsqueeze(0)
    ego_future_rot_tensor = torch.from_numpy(ego_future_rot_local).float().unsqueeze(0).unsqueeze(0)

    # Load camera images
    image_frames_list = []
    camera_indices_list = []
    timestamps_list = []

    # Image timestamps: if num_frames=4, load at [t0-0.3s, t0-0.2s, t0-0.1s, t0]
    image_timestamps = np.array(
        [t0_us - (num_frames - 1 - i) * int(time_step * 1_000_000) for i in range(num_frames)],
        dtype=np.int64,
    )

    for cam_feature in camera_features:
        camera = avdi.get_clip_feature(
            clip_id,
            cam_feature,
            maybe_stream=maybe_stream,
        )

        # frames: (num_frames, H, W, 3) uint8
        frames, frame_timestamps = camera.decode_images_from_timestamps(image_timestamps)

        # Convert to (num_frames, 3, H, W) for model input
        frames_tensor = torch.from_numpy(frames)
        frames_tensor = rearrange(frames_tensor, "t h w c -> t c h w")

        # Extract camera name from feature path
        cam_idx = camera_name_to_index[camera_features_by_name[cam_feature]]

        image_frames_list.append(frames_tensor)
        camera_indices_list.append(cam_idx)
        timestamps_list.append(torch.from_numpy(frame_timestamps.astype(np.int64)))

    # Stack and sort by camera index for consistent ordering
    image_frames = torch.stack(image_frames_list, dim=0)  # (N_cameras, num_frames, 3, H, W)
    camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)  # (N_cameras,)
    all_timestamps = torch.stack(timestamps_list, dim=0)  # (N_cameras, num_frames)

    # Sort by camera index to ensure consistent ordering [0, 1, 2, 6] instead of arbitrary order
    sort_order = torch.argsort(camera_indices)
    image_frames = image_frames[sort_order]
    camera_indices = camera_indices[sort_order]
    all_timestamps = all_timestamps[sort_order]

    # PhysicalAI-AV exposes image timestamps in the clip-relative time base.
    absolute_timestamps = all_timestamps
    camera_tmin_tensor = absolute_timestamps.min()
    camera_tmin = int(camera_tmin_tensor.item())
    relative_timestamps = (
        absolute_timestamps - camera_tmin_tensor
    ).float() * 1e-6  # (N_cameras, num_frames)
    ego_t0_us = int(t0_us)

    ego_history_tvals = torch.arange(
        -num_history_steps + 1,
        1,
        dtype=torch.float32,
    ) * float(time_step)
    ego_future_tvals = torch.arange(
        1,
        num_future_steps + 1,
        dtype=torch.float32,
    ) * float(time_step)
    t0_inv_quat_wxyz = np.array(
        [t0_quat[3], -t0_quat[0], -t0_quat[1], -t0_quat[2]],
        dtype=np.float32,
    )

    data = {
        "image_frames": image_frames,  # (N_cameras, num_frames, 3, H, W)
        "camera_indices": camera_indices,  # (N_cameras,)
        "camera_names": [camera_index_to_name[camera_id] for camera_id in camera_indices.tolist()],
        "ego_available": torch.tensor(True),
        "ego_t0": torch.tensor([ego_t0_us], dtype=torch.int64),
        "ego_t0_relative": torch.tensor([(ego_t0_us - camera_tmin) * 1e-6], dtype=torch.float32),
        "ego_t0_frame_idx": torch.tensor([num_frames - 1], dtype=torch.int64),
        "prediction_start_offset": torch.zeros(1, dtype=torch.float32),
        "ego_history_tvals": ego_history_tvals,
        "ego_history_xyz": ego_history_xyz_tensor,  # (1, 1, num_history_steps, 3)
        "ego_history_rot": ego_history_rot_tensor,  # (1, 1, num_history_steps, 3, 3)
        "ego_future_tvals": ego_future_tvals,
        "ego_future_xyz": ego_future_xyz_tensor,  # (1, 1, num_future_steps, 3)
        "ego_future_rot": ego_future_rot_tensor,  # (1, 1, num_future_steps, 3, 3)
        "relative_timestamps": relative_timestamps,  # (N_cameras, num_frames)
        "absolute_timestamps": absolute_timestamps,  # (N_cameras, num_frames)
        "camera_tmin": camera_tmin,
        "ego_t0_xyz": torch.from_numpy(t0_xyz).float().view(1, 1, 3),
        "ego_t0_inv_quat": torch.from_numpy(t0_inv_quat_wxyz).float().view(1, 1, 4),
        "t0_us": t0_us,
        "clip_id": clip_id,
    }
    if include_calibration:
        calibration = _load_camera_calibrations(
            avdi=avdi,
            clip_id=clip_id,
            camera_indices=camera_indices.tolist(),
            camera_index_to_name=camera_index_to_name,
            maybe_stream=maybe_stream,
        )
        if calibration:
            data["camera_calibrations"] = calibration
    return data


def _load_camera_calibrations(
    avdi: physical_ai_av.PhysicalAIAVDatasetInterface,
    clip_id: str,
    camera_indices: list[int],
    camera_index_to_name: dict[int, str],
    maybe_stream: bool,
) -> dict[int, dict[str, Any]]:
    """Load camera models and sensor poses keyed by Alpamayo camera index."""
    try:
        intrinsics = avdi.get_clip_feature(
            clip_id,
            "camera_intrinsics",
            maybe_stream=maybe_stream,
        )
        extrinsics = avdi.get_clip_feature(
            clip_id,
            "sensor_extrinsics",
            maybe_stream=maybe_stream,
        )
    except Exception as exc:  # pragma: no cover - depends on remote dataset availability.
        logger.warning("Camera calibration unavailable for %s: %s", clip_id, exc)
        return {}

    calibrations: dict[int, dict[str, Any]] = {}
    for camera_idx in camera_indices:
        camera_name = camera_index_to_name.get(camera_idx)
        if camera_name is None:
            continue
        camera_model = intrinsics.camera_models.get(camera_name)
        sensor_pose = extrinsics.sensor_poses.get(camera_name)
        if camera_model is None or sensor_pose is None:
            continue
        calibrations[camera_idx] = {
            "camera_name": camera_name,
            "camera_model": camera_model,
            "sensor_pose": sensor_pose,
        }
    return calibrations
