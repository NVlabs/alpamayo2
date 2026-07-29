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

"""Tests for PhysicalAI-AV sample loading used by release inference."""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from alpamayo2_super.inference_smoke import load_validation_sample
from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset


class _FakeRotation:
    def __init__(self, count: int) -> None:
        self._quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), (count, 1))

    def as_quat(self) -> np.ndarray:
        return self._quat


class _FakeEgomotion:
    def __call__(self, timestamps: np.ndarray) -> SimpleNamespace:
        seconds = timestamps.astype(np.float64) * 1e-6
        translation = np.stack([seconds, seconds * 0.5, seconds * 0.0], axis=-1)
        pose = SimpleNamespace(
            translation=translation,
            rotation=_FakeRotation(len(timestamps)),
        )
        return SimpleNamespace(pose=pose)


class _FakeCamera:
    def __init__(self, offset_us: int) -> None:
        self.offset_us = offset_us

    def decode_images_from_timestamps(
        self, image_timestamps: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        frames = np.zeros((len(image_timestamps), 2, 2, 3), dtype=np.uint8)
        for frame_idx in range(len(image_timestamps)):
            frames[frame_idx, :, :, :] = self.offset_us + frame_idx
        return frames, image_timestamps + self.offset_us


class _FakeFeatures:
    class LABELS:
        EGOMOTION = "labels/egomotion"

    class CAMERA:
        CAMERA_CROSS_LEFT_120FOV = "camera/camera_cross_left_120fov"
        CAMERA_FRONT_WIDE_120FOV = "camera/camera_front_wide_120fov"
        CAMERA_FRONT_TELE_30FOV = "camera/camera_front_tele_30fov"


class _FakeAVDI:
    features = _FakeFeatures()

    _camera_offsets = {
        _FakeFeatures.CAMERA.CAMERA_CROSS_LEFT_120FOV: 20,
        _FakeFeatures.CAMERA.CAMERA_FRONT_WIDE_120FOV: 30,
        _FakeFeatures.CAMERA.CAMERA_FRONT_TELE_30FOV: 60,
    }

    def get_clip_feature(
        self,
        clip_id: str,
        feature: str,
        maybe_stream: bool = True,
    ) -> _FakeCamera | _FakeEgomotion:
        del clip_id, maybe_stream
        if feature == self.features.LABELS.EGOMOTION:
            return _FakeEgomotion()
        return _FakeCamera(self._camera_offsets[feature])


def test_loader_rejects_unknown_camera_feature_name() -> None:
    """Unknown feature names must not silently become camera zero."""
    avdi = _FakeAVDI()
    unknown_feature = "camera/camera_unknown"
    avdi._camera_offsets[unknown_feature] = 10

    with pytest.raises(ValueError, match="unknown camera feature"):
        load_physical_aiavdataset(
            "clip",
            t0_us=500_000,
            avdi=avdi,
            camera_features=[unknown_feature],
            include_calibration=False,
        )


def test_loader_keeps_avdi_as_third_positional_argument() -> None:
    """The public loader should preserve its original positional API."""
    parameter_names = list(inspect.signature(load_physical_aiavdataset).parameters)

    assert parameter_names[:3] == ["clip_id", "t0_us", "avdi"]


def test_loader_reports_clip_relative_timebase() -> None:
    """Public timing metadata should use the clip-relative PhysicalAI-AV time base."""
    avdi = _FakeAVDI()

    data = load_physical_aiavdataset(
        "clip",
        t0_us=500_000,
        avdi=avdi,
        num_history_steps=3,
        num_future_steps=2,
        time_step=0.1,
        camera_features=[
            avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
            avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
            avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
        ],
        num_frames=2,
        include_calibration=False,
    )

    assert data["camera_indices"].tolist() == [0, 1, 6]
    assert data["absolute_timestamps"].tolist() == [
        [400_020, 500_020],
        [400_030, 500_030],
        [400_060, 500_060],
    ]
    assert data["camera_tmin"] == 400_020
    torch.testing.assert_close(
        data["relative_timestamps"],
        torch.tensor(
            [
                [0.0, 0.1],
                [0.00001, 0.10001],
                [0.00004, 0.10004],
            ],
            dtype=torch.float32,
        ),
    )
    assert data["ego_t0"].tolist() == [500_000]
    torch.testing.assert_close(data["ego_t0_relative"], torch.tensor([0.09998]))
    assert data["ego_t0_frame_idx"].tolist() == [1]
    torch.testing.assert_close(data["prediction_start_offset"], torch.zeros(1))
    torch.testing.assert_close(data["ego_history_tvals"], torch.tensor([-0.2, -0.1, 0.0]))
    torch.testing.assert_close(data["ego_future_tvals"], torch.tensor([0.1, 0.2]))
    torch.testing.assert_close(data["ego_t0_xyz"], torch.tensor([[[0.5, 0.25, 0.0]]]))
    torch.testing.assert_close(data["ego_t0_inv_quat"], torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]))
    assert "clip_start_timestamp_us" not in data


def test_validation_manifest_ignores_internal_absolute_timing_metadata(tmp_path) -> None:
    """Public inference only consumes clip-relative sample identifiers."""
    manifest_path = tmp_path / "samples.json"
    manifest_path.write_text(
        """
        {
          "samples": [
            {
              "clip_id": "clip",
              "t0_us": 500000,
              "clip_start_timestamp_us": 1000000,
              "event_t0_us_abs": 1500000
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    assert load_validation_sample(manifest_path, 0) == {
        "clip_id": "clip",
        "t0_us": 500_000,
    }
