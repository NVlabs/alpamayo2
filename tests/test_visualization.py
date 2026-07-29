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

"""Tests for inference visualization metadata."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import torch
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba

from alpamayo2_super.visualization import (
    plot_blog_figure,
    plot_auto_labeling_result,
    plot_compact_inference_result,
    plot_grounding_result,
    plot_inference_result,
    plot_meta_action_result,
    plot_vqa_result,
)


def test_plot_compact_inference_result_writes_shape_and_timing_metadata(tmp_path: Path) -> None:
    """The compact JSON sidecar includes enough context to inspect debug outputs."""
    data = {
        "clip_id": "clip",
        "t0_us": 500_000,
        "camera_tmin": torch.tensor(1_400_000),
        "camera_indices": torch.tensor([1]),
        "image_frames": torch.zeros((1, 1, 3, 8, 8), dtype=torch.uint8),
        "absolute_timestamps": torch.tensor([[1_400_000]], dtype=torch.int64),
        "relative_timestamps": torch.tensor([[0.0]], dtype=torch.float32),
        "ego_history_xyz": torch.zeros((1, 1, 2, 3), dtype=torch.float32),
        "ego_future_xyz": torch.zeros((1, 1, 2, 3), dtype=torch.float32),
    }
    pred_xyz = torch.zeros((1, 1, 1, 2, 3), dtype=torch.float32)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text(
        '{"model_type": "alpamayo2_super"}\n',
        encoding="utf-8",
    )
    (checkpoint_dir / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    (checkpoint_dir / "00000.safetensors").write_bytes(b"weights")
    (checkpoint_dir / "00001.safetensors").symlink_to(checkpoint_dir / "00000.safetensors")
    json_path = tmp_path / "sample.json"

    fig, metadata = plot_compact_inference_result(
        data=data,
        pred_xyz=pred_xyz,
        extra={"cot": ["Proceed straight."]},
        json_path=json_path,
        model_id=str(checkpoint_dir),
    )
    plt.close(fig)

    assert metadata["camera_tmin"] == 1_400_000
    assert metadata["camera_indices"] == [1]
    assert metadata["pred_xyz_shape"] == [1, 1, 1, 2, 3]
    assert metadata["image_frames_shape"] == [1, 1, 3, 8, 8]
    assert metadata["ego_future_xyz_shape"] == [1, 1, 2, 3]
    assert metadata["checkpoint_file_count"] == 4
    assert metadata["checkpoint_safetensors_count"] == 2
    assert metadata["checkpoint_has_symlinks"] is True
    assert len(metadata["checkpoint_config_sha256"]) == 64
    assert len(metadata["checkpoint_index_sha256"]) == 64
    assert json.loads(json_path.read_text(encoding="utf-8")) == metadata


def test_plot_inference_result_writes_task_native_driving_layout(tmp_path: Path) -> None:
    """The default figure uses the selected six-camera driving profile without placeholders."""
    data = {
        "clip_id": "030c760c-ae38-49aa-9ad8-f5650a545d26",
        "t0_us": 5_100_000,
        "camera_tmin": torch.tensor(4_800_000),
        "camera_indices": torch.tensor([0, 1, 2, 3, 5, 6]),
        "image_frames": torch.zeros((6, 1, 3, 8, 8), dtype=torch.uint8),
        "absolute_timestamps": torch.full((6, 1), 4_800_000, dtype=torch.int64),
        "relative_timestamps": torch.zeros((6, 1), dtype=torch.float32),
        "ego_history_xyz": torch.zeros((1, 1, 2, 3), dtype=torch.float32),
        "ego_future_xyz": torch.zeros((1, 1, 2, 3), dtype=torch.float32),
    }
    pred_xyz = torch.zeros((1, 1, 1, 2, 3), dtype=torch.float32)
    output_path = tmp_path / "blog.png"
    json_path = tmp_path / "blog.json"

    fig, metadata = plot_inference_result(
        data=data,
        pred_xyz=pred_xyz,
        extra={"cot": ["Keep distance to the lead vehicle."]},
        output_path=output_path,
        json_path=json_path,
        model_id="nvidia/Alpamayo2-Super",
        seed=42,
    )
    camera_positions = {
        ax.get_title(): (
            ax.get_subplotspec().rowspan.start,
            ax.get_subplotspec().colspan.start,
        )
        for ax in fig.axes
        if ax.get_title() in CAMERA_TITLES
    }
    plt.close(fig)

    assert output_path.is_file()
    assert metadata["figure_style"] == "blog_6cam_task_native"
    assert metadata["clip_id"] == "030c760c-ae38-49aa-9ad8-f5650a545d26"
    assert metadata["t0_us"] == 5_100_000
    assert metadata["camera_grid_camera_ids"] == [0, 1, 2, 3, 6, 5]
    assert metadata["camera_titles"] == [
        "cross left",
        "front wide",
        "cross right",
        "rear left",
        "front tele",
        "rear right",
    ]
    assert camera_positions == {
        "cross left": (0, 0),
        "front wide": (0, 1),
        "cross right": (0, 2),
        "rear left": (1, 0),
        "front tele": (1, 1),
        "rear right": (1, 2),
    }
    assert metadata["projection_available"] is False
    assert metadata["cot"] == "Keep distance to the lead vehicle."
    assert metadata["coc_label"] == "Predicted CoC"
    assert metadata["camera_ribbon_width_m"] == 2.0
    assert metadata["camera_ribbon_semantics"] == "approximate vehicle-width corridor"
    assert metadata["prediction_layer"] == "above_ground_truth"
    assert metadata["trajectory_palette"] == {
        "prediction": "#76B900",
        "ground_truth": "#EE3377",
        "history": "#0077BB",
        "ego": "#EE7733",
    }
    assert json.loads(json_path.read_text(encoding="utf-8")) == metadata


class _ForwardCameraPose:
    """Map ego x-forward coordinates to camera z-forward coordinates."""

    def inv(self):
        """Return the inverse pose used by the projection helper."""
        return self

    def apply(self, xyz: np.ndarray) -> np.ndarray:
        """Transform ego XYZ into the synthetic camera frame."""
        return np.column_stack((xyz[:, 1], -xyz[:, 2], xyz[:, 0]))


class _PinholeCamera:
    """Project synthetic camera rays into a 200-by-200 image."""

    def ray2pixel(self, xyz: np.ndarray) -> np.ndarray:
        """Apply a 100-pixel focal length and centered principal point."""
        return np.column_stack(
            (
                100.0 + 100.0 * xyz[:, 0] / xyz[:, 2],
                100.0 + 100.0 * xyz[:, 1] / xyz[:, 2],
            )
        )


class _IdentityCameraPose:
    """Keep synthetic trajectory coordinates in the camera frame."""

    def inv(self):
        """Return the inverse pose used by the projection helper."""
        return self

    def apply(self, xyz: np.ndarray) -> np.ndarray:
        """Return camera-frame XYZ without changing coordinates."""
        return xyz


class _DirectCamera:
    """Treat synthetic camera x/y coordinates as image pixels."""

    def ray2pixel(self, xyz: np.ndarray) -> np.ndarray:
        """Return the first two camera coordinates as image coordinates."""
        return xyz[:, :2]


def _calibrated_front_wide_sample(
    trajectory: torch.Tensor,
    sensor_pose=None,
    camera_model=None,
) -> dict:
    """Build a driving-profile sample with synthetic front-wide calibration."""
    camera_ids = [0, 1, 2, 3, 5, 6]
    return {
        "clip_id": "clip",
        "t0_us": 500_000,
        "camera_tmin": torch.tensor(400_000),
        "camera_indices": torch.tensor(camera_ids),
        "image_frames": torch.zeros((6, 1, 3, 200, 200), dtype=torch.uint8),
        "absolute_timestamps": torch.full((6, 1), 400_000, dtype=torch.int64),
        "relative_timestamps": torch.zeros((6, 1), dtype=torch.float32),
        "ego_history_xyz": torch.zeros((1, 1, 2, 3), dtype=torch.float32),
        "ego_future_xyz": trajectory.unsqueeze(0),
        "camera_calibrations": {
            1: {
                "sensor_pose": sensor_pose or _ForwardCameraPose(),
                "camera_model": camera_model or _PinholeCamera(),
            }
        },
    }


def test_plot_inference_result_projects_two_meter_vehicle_width_ribbons() -> None:
    """Calibrated overlays use physical two-meter ribbons with prediction above GT."""
    trajectory = torch.tensor(
        [[[5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [15.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    data = _calibrated_front_wide_sample(trajectory)
    pred_xyz = trajectory.unsqueeze(0).unsqueeze(0)

    fig, metadata = plot_inference_result(data=data, pred_xyz=pred_xyz)
    front_wide_axis = next(ax for ax in fig.axes if ax.get_title() == "front wide")
    ribbons = [
        collection
        for collection in front_wide_axis.collections
        if isinstance(collection, PolyCollection)
    ]

    assert metadata["projected_camera_ids"] == [1]
    assert len(ribbons) == 2
    ground_truth_ribbon, prediction_ribbon = ribbons
    assert np.allclose(ground_truth_ribbon.get_facecolor()[0], to_rgba("#EE3377", alpha=0.10))
    assert np.allclose(prediction_ribbon.get_facecolor()[0], to_rgba("#76B900", alpha=0.18))
    assert prediction_ribbon.get_zorder() > ground_truth_ribbon.get_zorder()
    assert np.isclose(np.ptp(prediction_ribbon.get_paths()[0].vertices[:, 0]), 40.0)

    ground_truth_line, prediction_line = front_wide_axis.lines
    assert ground_truth_line.get_color() == "#EE3377"
    assert ground_truth_line.get_linestyle() == "--"
    assert prediction_line.get_color() == "#76B900"
    assert prediction_line.get_linestyle() == "-"
    assert prediction_line.get_zorder() > ground_truth_line.get_zorder()
    plt.close(fig)


def test_plot_inference_result_preserves_ribbon_width_for_stationary_path() -> None:
    """A repeated stationary path retains a lateral vehicle-width normal."""
    trajectory = torch.tensor(
        [[[5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    data = _calibrated_front_wide_sample(trajectory)

    fig, _ = plot_inference_result(
        data=data,
        pred_xyz=trajectory.unsqueeze(0).unsqueeze(0),
    )
    front_wide_axis = next(ax for ax in fig.axes if ax.get_title() == "front wide")
    prediction_ribbon = front_wide_axis.collections[1]

    assert np.isclose(np.ptp(prediction_ribbon.get_paths()[0].vertices[:, 0]), 40.0)
    plt.close(fig)


def test_plot_inference_result_accepts_single_point_trajectory() -> None:
    """A one-point path does not fail while constructing its ground-plane normal."""
    trajectory = torch.tensor([[[5.0, 0.0, 0.0]]], dtype=torch.float32)
    data = _calibrated_front_wide_sample(trajectory)

    fig, metadata = plot_inference_result(
        data=data,
        pred_xyz=trajectory.unsqueeze(0).unsqueeze(0),
    )

    assert metadata["projection_available"] is False
    plt.close(fig)


def test_plot_inference_result_clips_partially_visible_ribbons() -> None:
    """A corridor edge outside the frame is clipped instead of dropping the segment."""
    trajectory = torch.tensor(
        [[[2.0, 1.5, 0.0], [3.0, 1.5, 0.0]]],
        dtype=torch.float32,
    )
    data = _calibrated_front_wide_sample(trajectory)

    fig, metadata = plot_inference_result(
        data=data,
        pred_xyz=trajectory.unsqueeze(0).unsqueeze(0),
    )
    front_wide_axis = next(ax for ax in fig.axes if ax.get_title() == "front wide")
    ribbons = [
        collection
        for collection in front_wide_axis.collections
        if isinstance(collection, PolyCollection)
    ]

    assert metadata["projected_camera_ids"] == [1]
    assert len(ribbons) == 2
    vertices = ribbons[1].get_paths()[0].vertices
    assert np.all(vertices[:, 0] >= -0.5)
    assert np.all(vertices[:, 0] <= 199.5)
    assert np.all(vertices[:, 1] >= -0.5)
    assert np.all(vertices[:, 1] <= 199.5)
    assert ribbons[1].get_clip_on() is True
    assert np.allclose(front_wide_axis.get_xlim(), (-0.5, 199.5))
    assert np.allclose(front_wide_axis.get_ylim(), (199.5, -0.5))
    plt.close(fig)


def test_plot_inference_result_clips_segments_crossing_camera_near_plane() -> None:
    """A path entering the camera frustum remains visible after near-plane clipping."""
    trajectory = torch.tensor(
        [[[-1.0, -2.0, 0.0], [5.0, 2.0, 0.0]]],
        dtype=torch.float32,
    )
    data = _calibrated_front_wide_sample(trajectory)

    fig, metadata = plot_inference_result(
        data=data,
        pred_xyz=trajectory.unsqueeze(0).unsqueeze(0),
    )
    front_wide_axis = next(ax for ax in fig.axes if ax.get_title() == "front wide")
    ribbons = [
        collection
        for collection in front_wide_axis.collections
        if isinstance(collection, PolyCollection)
    ]

    assert metadata["projected_camera_ids"] == [1]
    assert len(ribbons) == 2
    assert len(front_wide_axis.lines) == 2
    plt.close(fig)


def test_plot_inference_result_rejects_projection_outside_image_corner() -> None:
    """An off-corner strip with an overlapping AABB is not reported as visible."""
    trajectory = torch.tensor(
        [[[-10.0, 5.0, 1.0], [5.0, -10.0, 1.0]]],
        dtype=torch.float32,
    )
    data = _calibrated_front_wide_sample(
        trajectory,
        sensor_pose=_IdentityCameraPose(),
        camera_model=_DirectCamera(),
    )

    fig, metadata = plot_inference_result(
        data=data,
        pred_xyz=trajectory.unsqueeze(0).unsqueeze(0),
    )
    front_wide_axis = next(ax for ax in fig.axes if ax.get_title() == "front wide")
    ribbons = [
        collection
        for collection in front_wide_axis.collections
        if isinstance(collection, PolyCollection)
    ]

    assert metadata["projection_available"] is False
    assert metadata["projected_camera_ids"] == []
    assert ribbons == []
    assert len(front_wide_axis.lines) == 0
    plt.close(fig)


def test_plot_inference_result_does_not_report_fully_offscreen_ribbon() -> None:
    """A finite but offscreen projection remains unavailable in metadata."""
    trajectory = torch.tensor(
        [[[2.0, 10.0, 0.0], [3.0, 10.0, 0.0]]],
        dtype=torch.float32,
    )
    data = _calibrated_front_wide_sample(trajectory)

    fig, metadata = plot_inference_result(
        data=data,
        pred_xyz=trajectory.unsqueeze(0).unsqueeze(0),
    )

    assert metadata["projection_available"] is False
    assert metadata["projected_camera_ids"] == []
    plt.close(fig)


def test_plot_inference_result_uses_release_trajectory_palette_in_bev() -> None:
    """The BEV uses distinct prediction, GT, history, and ego colors."""
    data = _driving_six_camera_sample()
    pred_xyz = torch.zeros((1, 1, 1, 2, 3), dtype=torch.float32)

    fig, _ = plot_inference_result(data=data, pred_xyz=pred_xyz)
    bev_axis = next(
        ax for ax in fig.axes if ax.get_title() == "Predicted Trajectory vs Ground Truth Motion"
    )
    lines = {line.get_label(): line for line in bev_axis.lines}
    ego_collection = next(
        collection for collection in bev_axis.collections if collection.get_label() == "ego"
    )

    assert lines["pred"].get_color() == "#76B900"
    assert lines["gt"].get_color() == "#EE3377"
    assert lines["history"].get_color() == "#0077BB"
    assert lines["pred"].get_zorder() > lines["gt"].get_zorder()
    assert np.allclose(ego_collection.get_facecolor()[0], to_rgba("#EE7733"))
    plt.close(fig)


def test_plot_inference_result_records_all_cots_and_unambiguous_metrics(
    tmp_path: Path,
) -> None:
    """Multi-sample metadata should distinguish independent and paired minima."""
    data = {
        "clip_id": "clip",
        "t0_us": 500_000,
        "camera_tmin": torch.tensor(400_000),
        "camera_indices": torch.tensor([1]),
        "image_frames": torch.zeros((1, 1, 3, 8, 8), dtype=torch.uint8),
        "absolute_timestamps": torch.tensor([[400_000]], dtype=torch.int64),
        "relative_timestamps": torch.zeros((1, 1), dtype=torch.float32),
        "ego_history_xyz": torch.zeros((1, 1, 2, 3), dtype=torch.float32),
        "ego_future_xyz": torch.zeros((1, 1, 2, 3), dtype=torch.float32),
    }
    pred_xyz = torch.zeros((1, 1, 2, 2, 3), dtype=torch.float32)
    pred_xyz[0, 0, 0, :, 0] = torch.tensor([0.0, 4.0])
    pred_xyz[0, 0, 1, :, 0] = torch.tensor([3.0, 3.0])

    fig, metadata = plot_compact_inference_result(
        data=data,
        pred_xyz=pred_xyz,
        extra={"cot": [["first explanation", "second explanation"]]},
        json_path=tmp_path / "multi_sample.json",
    )
    plt.close(fig)

    assert metadata["cot"] == "first explanation"
    assert metadata["cots"] == ["first explanation", "second explanation"]
    assert metadata["ade_m"] == [2.0, 3.0]
    assert metadata["fde_m"] == [4.0, 3.0]
    assert metadata["min_ade_m"] == 2.0
    assert metadata["min_fde_m"] == 3.0
    assert metadata["fde_at_min_ade_m"] == 4.0
    assert metadata["best_ade_sample_index"] == 0
    assert metadata["best_fde_sample_index"] == 1
    assert "best_sample_index" not in metadata


def test_plot_blog_figure_matches_default_public_renderer(tmp_path: Path) -> None:
    """The old blog helper name stays an alias for the default public renderer."""
    data = _driving_six_camera_sample()
    pred_xyz = torch.zeros((1, 1, 1, 2, 3), dtype=torch.float32)

    fig, default_metadata = plot_inference_result(
        data=data,
        pred_xyz=pred_xyz,
        extra={"cot": ["Keep distance to the lead vehicle."]},
        output_path=tmp_path / "default.png",
        model_id="nvidia/Alpamayo2-Super",
        seed=42,
    )
    plt.close(fig)
    fig, alias_metadata = plot_blog_figure(
        data=data,
        pred_xyz=pred_xyz,
        extra={"cot": ["Keep distance to the lead vehicle."]},
        output_path=tmp_path / "alias.png",
        model_id="nvidia/Alpamayo2-Super",
        seed=42,
    )
    plt.close(fig)

    assert alias_metadata == default_metadata


def _camera_sample(camera_ids: list[int], num_frames: int = 1) -> dict:
    """Build a tiny task-profile sample for blog-style visualization tests."""
    camera_count = len(camera_ids)
    return {
        "clip_id": "030c760c-ae38-49aa-9ad8-f5650a545d26",
        "t0_us": 5_100_000,
        "camera_tmin": torch.tensor(4_800_000),
        "camera_indices": torch.tensor(camera_ids),
        "image_frames": torch.zeros((camera_count, num_frames, 3, 8, 8), dtype=torch.uint8),
        "absolute_timestamps": torch.full((camera_count, num_frames), 4_800_000, dtype=torch.int64),
        "relative_timestamps": torch.zeros((camera_count, num_frames), dtype=torch.float32),
        "ego_history_xyz": torch.zeros((1, 1, 2, 3), dtype=torch.float32),
        "ego_future_xyz": torch.zeros((1, 1, 2, 3), dtype=torch.float32),
        "input_profile": {
            "camera_ids": camera_ids,
            "frame_indices": list(range(num_frames)),
            "num_cameras": camera_count,
            "num_frames_per_camera": num_frames,
        },
    }


def _driving_six_camera_sample(num_frames: int = 1) -> dict:
    """Build the validated trajectory/meta/auto/grounding camera profile."""
    return _camera_sample([0, 1, 2, 3, 5, 6], num_frames=num_frames)


def _vqa_six_camera_sample(num_frames: int = 1) -> dict:
    """Build the validated VQA camera profile."""
    return _camera_sample([0, 1, 2, 3, 4, 5], num_frames=num_frames)


CAMERA_TITLES = {
    "cross left",
    "front wide",
    "cross right",
    "rear left",
    "rear tele",
    "rear right",
    "front tele",
}


def _assert_panel_text_is_contained(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    """Assert every text artist stays inside its panel border."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        panel_bounds = ax.get_window_extent(renderer)
        for text in ax.texts:
            text_bounds = text.get_window_extent(renderer)
            assert text_bounds.x0 >= panel_bounds.x0 - 1
            assert text_bounds.x1 <= panel_bounds.x1 + 1
            assert text_bounds.y0 >= panel_bounds.y0 - 1
            assert text_bounds.y1 <= panel_bounds.y1 + 1


def test_plot_meta_action_result_writes_blog_style_metadata(tmp_path: Path) -> None:
    """The meta-action figure shows its six-camera task profile and parsed action axes."""
    output_path = tmp_path / "meta_action.png"

    fig, metadata = plot_meta_action_result(
        data=_driving_six_camera_sample(),
        cot="Slow for the lead vehicle.",
        meta_action=("Longitudinal: Decelerate.\nLateral: Go Straight.\nLane: Lane Keep."),
        output_path=output_path,
        model_id="nvidia/Alpamayo2-Super",
        seed=42,
    )
    axes_count = len(fig.axes)

    assert output_path.is_file()
    assert metadata["task"] == "meta_action"
    assert metadata["figure_style"] == "blog_meta_action"
    assert axes_count == 7
    assert metadata["camera_grid_camera_ids"] == [0, 1, 2, 3, 6, 5]
    assert metadata["camera_indices"] == [0, 1, 2, 3, 5, 6]
    assert metadata["cot"] == "Slow for the lead vehicle."
    assert metadata["meta_action_fields"] == {
        "Longitudinal": "Decelerate.",
        "Lateral": "Go Straight.",
        "Lane": "Lane Keep.",
    }
    assert metadata["image_frames_shape"] == [6, 1, 3, 8, 8]
    _assert_panel_text_is_contained(fig, fig.axes[-1:])
    plt.close(fig)


def test_plot_vqa_result_writes_blog_style_metadata(tmp_path: Path) -> None:
    """The VQA figure shows its six-camera task profile plus question and answer text."""
    output_path = tmp_path / "vqa.png"

    fig, metadata = plot_vqa_result(
        data=_vqa_six_camera_sample(),
        question=(
            "What are the key traffic elements visible in this scene and how should they "
            "influence driving behavior?"
        ),
        answer="The ego vehicle should pay attention to the lead car and work-zone cones.",
        output_path=output_path,
        model_id="nvidia/Alpamayo2-Super",
        seed=42,
    )

    assert output_path.is_file()
    assert metadata["task"] == "vqa"
    assert metadata["figure_style"] == "blog_vqa"
    assert metadata["question"].startswith("What are the key traffic elements")
    assert metadata["answer"].startswith("The ego vehicle should pay attention")
    assert "grounding_text" not in metadata
    assert metadata["camera_grid_camera_ids"] == [0, 1, 2, 3, 4, 5]
    assert metadata["camera_indices"] == [0, 1, 2, 3, 4, 5]
    camera_positions = {
        ax.get_title(): (
            ax.get_subplotspec().rowspan.start,
            ax.get_subplotspec().colspan.start,
        )
        for ax in fig.axes
        if ax.get_title() in CAMERA_TITLES
    }
    assert camera_positions["front wide"] == (0, 1)
    assert camera_positions["rear tele"] == (1, 1)
    _assert_panel_text_is_contained(fig, fig.axes[-2:])
    plt.close(fig)


def test_plot_grounding_result_overlays_grounding_boxes(tmp_path: Path) -> None:
    """Grounding JSON is parsed into boxes overlaid on selected camera panels."""
    output_path = tmp_path / "grounding.png"
    grounding_text = '[{"bbox_2d": [0.1, 0.2, 0.6, 0.7], "label": "lead vehicle"}]'

    fig, metadata = plot_grounding_result(
        data=_driving_six_camera_sample(),
        question=(
            "Find the white lead vehicle directly ahead and return its bounding box in JSON."
        ),
        answer=grounding_text,
        grounding_text=grounding_text,
        grounding_camera_ids=(1,),
        output_path=output_path,
        model_id="nvidia/Alpamayo2-Super",
        seed=42,
    )
    assert output_path.is_file()
    assert metadata["task"] == "grounding"
    assert metadata["figure_style"] == "blog_grounding"
    assert metadata["grounding_box_count"] == 1
    assert metadata["grounding_camera_ids"] == [1]
    assert metadata["grounding_boxes"] == [{"bbox": [0.1, 0.2, 0.6, 0.7], "label": "lead vehicle"}]
    assert "\n" not in fig.axes[-2].texts[-1].get_text()
    assert "bbox_2d: [0.1, 0.2, 0.6, 0.7]" in fig.axes[-1].texts[-1].get_text()
    _assert_panel_text_is_contained(fig, fig.axes[-2:])
    plt.close(fig)


def test_plot_grounding_result_salvages_truncated_grounding_json(tmp_path: Path) -> None:
    """Complete bbox entries are visualized even if the generated JSON list is truncated."""
    output_path = tmp_path / "grounding_truncated.png"
    grounding_text = '[{"bbox_2d": [428, 486, 517, 602], "label": "Car"}'

    fig, metadata = plot_grounding_result(
        data=_driving_six_camera_sample(),
        question="Locate the lead vehicle.",
        answer=grounding_text,
        grounding_text=grounding_text,
        output_path=output_path,
        model_id="nvidia/Alpamayo2-Super",
        seed=42,
    )
    plt.close(fig)

    assert output_path.is_file()
    assert metadata["grounding_box_count"] == 1
    assert metadata["grounding_camera_ids"] == [1]
    assert metadata["grounding_boxes"] == [{"bbox": [428.0, 486.0, 517.0, 602.0], "label": "Car"}]


def test_plot_auto_labeling_result_writes_video_and_poster(tmp_path: Path) -> None:
    """The auto-labeling visual animates every model-input frame and writes a poster."""
    poster_path = tmp_path / "auto_labeling_poster.png"
    video_path = tmp_path / "auto_labeling.mp4"
    auto_labeling_json = {
        "critical_components_analysis": "Lead vehicle and cones constrain the ego path.",
        "ego_vehicle_motion_analysis": (
            "1. type: keep lane,\n"
            "duration: 0-2 seconds,\n"
            "motion: follow the lead vehicle.\n"
            "2. type: nudge left,\n"
            "duration: 2-6 seconds,\n"
            "motion: increase clearance from the roadwork equipment.\n"
            "3. type: keep lane,\n"
            "duration: 6-8 seconds,\n"
            "motion: maintain distance behind the lead vehicle."
        ),
        "trajectory_analysis": "The trajectory stays centered and follows the lead vehicle.",
        "chain_of_causation": "Ego keeps distance because the lead vehicle sets speed.",
    }
    data = _driving_six_camera_sample(num_frames=4)
    for frame_index in range(4):
        data["image_frames"][:, frame_index].fill_(frame_index * 70)

    fig, metadata = plot_auto_labeling_result(
        data=data,
        auto_labeling_json=auto_labeling_json,
        auto_labeling_text="Generated auto-labeling JSON.",
        future_source="ground_truth",
        output_path=poster_path,
        video_path=video_path,
        model_id="nvidia/Alpamayo2-Super",
        seed=42,
    )
    axes_count = len(fig.axes)

    assert poster_path.is_file()
    assert video_path.is_file()
    video = media.read_video(video_path)
    assert video.shape[0] == 4
    assert np.mean(np.abs(video[0].astype(float) - video[-1].astype(float))) > 1.0
    assert metadata["task"] == "auto_labeling"
    assert metadata["figure_style"] == "blog_auto_labeling_video"
    assert metadata["input_mode"] == "multi_camera_context_video"
    assert metadata["num_cameras"] == 6
    assert metadata["num_frames_per_camera"] == 4
    assert metadata["video_frame_count"] == 4
    assert metadata["video_fps"] == 2.0
    assert metadata["video_path"] == str(video_path)
    assert metadata["poster_path"] == str(poster_path)
    assert metadata["image_frames_shape"] == [6, 4, 3, 8, 8]
    assert metadata["auto_labeling_json"] == auto_labeling_json
    assert metadata["future_source"] == "ground_truth"
    assert axes_count == 10
    assert min(ax.texts[-1].get_fontsize() for ax in fig.axes[-4:]) >= 16
    _assert_panel_text_is_contained(fig, fig.axes[-4:])
    plt.close(fig)
