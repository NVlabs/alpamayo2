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

"""Visualization utilities for Alpamayo 2 Super inference outputs."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import torch
from matplotlib.collections import PolyCollection
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox

from alpamayo2_super.input_profiles import (
    DRIVING_SIX_CAMERA_FOUR_FRAME,
    VQA_SIX_CAMERA_FOUR_FRAME,
)

CAMERA_GRID_LAYOUT = {
    6: (0, 1),  # front_tele
    0: (1, 0),  # cross_left
    1: (1, 1),  # front_wide
    2: (1, 2),  # cross_right
}
DRIVING_CAMERA_GRID_LAYOUT = {
    0: (0, 0),  # cross_left
    1: (0, 1),  # front_wide
    2: (0, 2),  # cross_right
    3: (1, 0),  # rear_left
    6: (1, 1),  # front_tele
    5: (1, 2),  # rear_right
}
VQA_CAMERA_GRID_LAYOUT = {
    0: (0, 0),  # cross_left
    1: (0, 1),  # front_wide
    2: (0, 2),  # cross_right
    3: (1, 0),  # rear_left
    4: (1, 1),  # rear_tele
    5: (1, 2),  # rear_right
}
DEFAULT_OVERLAY_CAMERA_IDS = (6, 0, 1, 2)
MIN_BEV_AXIS_RANGE_M = 8.0
MIN_BEV_LATERAL_RATIO = 0.25
CAMERA_NEAR_PLANE_M = 1e-3
TRAJECTORY_RIBBON_WIDTH_M = 2.0
TRAJECTORY_PREDICTION_COLOR = "#76B900"
TRAJECTORY_GROUND_TRUTH_COLOR = "#EE3377"
TRAJECTORY_HISTORY_COLOR = "#0077BB"
TRAJECTORY_EGO_COLOR = "#EE7733"
CAMERA_TITLES = {
    0: "cross left",
    1: "front wide",
    2: "cross right",
    3: "rear left",
    4: "rear tele",
    5: "rear right",
    6: "front tele",
}
BLOG_COC_LABEL = "Predicted CoC"
AUTO_LABELING_KEYS = (
    "critical_components_analysis",
    "ego_vehicle_motion_analysis",
    "trajectory_analysis",
    "chain_of_causation",
)
META_ACTION_FIELDS = ("Longitudinal", "Lateral", "Lane")
GROUNDING_BBOX_FALLBACK_RE = re.compile(
    r'"(?:bbox_2d|bbox|box)"\s*:\s*\[\s*'
    r"(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*\]"
    r'(?:\s*,\s*"label"\s*:\s*"([^"]*)")?'
)
BLOG_STYLE = {
    "camera_title_fontsize": 17,
    "bev_title_fontsize": 26,
    "coc_title_fontsize": 32,
    "coc_body_fontsize": 27,
    "bev_legend_fontsize": 18,
    "coc_body_y": 0.30,
    "coc_wrap_width": 128,
    "task_title_fontsize": 31,
    "task_body_fontsize": 24,
    "task_wrap_width": 128,
}


@dataclass(frozen=True)
class _TextPanel:
    """Text artist and constraints used for renderer-aware panel fitting."""

    ax: plt.Axes
    body_artist: Any
    body: str
    max_fontsize: int
    min_fontsize: int
    single_line: bool


def get_trajectories_xy(pred_xyz: torch.Tensor) -> np.ndarray:
    """Extract ``[K, T, 2]`` XY trajectories from model output."""
    return pred_xyz.detach().cpu().numpy()[0, 0, :, :, :2]


def _tensor_to_image(frame: torch.Tensor) -> np.ndarray:
    """Convert ``[3, H, W]`` tensor to displayable ``[H, W, 3]`` uint8 image."""
    image = frame.detach().cpu()
    if image.dtype != torch.uint8:
        image = torch.clamp(image, 0, 1) * 255
    return image.to(torch.uint8).permute(1, 2, 0).numpy()


def _extract_cots(extra: dict[str, Any] | None) -> list[str]:
    """Return every generated CoT string from an inference ``extra`` dict."""
    if not extra or "cot" not in extra:
        return []
    cot = np.asarray(extra["cot"], dtype=object)
    if cot.size == 0:
        return []
    return [str(value) for value in cot.reshape(-1)]


def _compute_metrics(pred_xyz: torch.Tensor, gt_future_xyz: torch.Tensor | None) -> dict[str, Any]:
    """Compute ADE/FDE diagnostics when ground truth is available."""
    if gt_future_xyz is None:
        return {
            "ade_m": [],
            "fde_m": [],
            "min_ade_m": None,
            "min_fde_m": None,
            "fde_at_min_ade_m": None,
            "best_ade_sample_index": None,
            "best_fde_sample_index": None,
        }

    pred_xy = get_trajectories_xy(pred_xyz)
    gt_xy = gt_future_xyz.detach().cpu().numpy()[0, 0, :, :2]
    distances = np.linalg.norm(pred_xy - gt_xy[None, :, :], axis=-1)
    ade = distances.mean(axis=-1)
    fde = distances[:, -1]
    best_ade_idx = int(np.argmin(ade))
    best_fde_idx = int(np.argmin(fde))
    return {
        "ade_m": ade.tolist(),
        "fde_m": fde.tolist(),
        "min_ade_m": float(ade[best_ade_idx]),
        "min_fde_m": float(fde[best_fde_idx]),
        "fde_at_min_ade_m": float(fde[best_ade_idx]),
        "best_ade_sample_index": best_ade_idx,
        "best_fde_sample_index": best_fde_idx,
    }


def _shape(value: Any) -> list[int] | None:
    """Return a JSON-friendly tensor shape."""
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    return None


def _json_scalar(value: Any) -> Any:
    """Return a JSON-friendly scalar."""
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha256_file(path: Path) -> str:
    """Return the SHA256 hash of a file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_metadata(model_id: str | None) -> dict[str, Any]:
    """Return lightweight provenance for local checkpoint directories."""
    if not model_id:
        return {}
    checkpoint_path = Path(model_id).expanduser()
    if not checkpoint_path.is_dir():
        return {}

    release_files = [
        path for path in checkpoint_path.iterdir() if path.is_file() or path.is_symlink()
    ]
    metadata: dict[str, Any] = {
        "checkpoint_file_count": len(release_files),
        "checkpoint_safetensors_count": len(list(checkpoint_path.glob("*.safetensors"))),
        "checkpoint_has_symlinks": any(path.is_symlink() for path in release_files),
    }
    for filename, key in [
        ("config.json", "checkpoint_config_sha256"),
        ("model.safetensors.index.json", "checkpoint_index_sha256"),
    ]:
        path = checkpoint_path / filename
        if path.is_file():
            metadata[key] = _sha256_file(path)
    return metadata


def _camera_coordinates(
    traj_xyz: np.ndarray,
    calibration: dict[str, Any],
) -> np.ndarray:
    """Transform an ego-frame trajectory into camera coordinates."""
    return np.asarray(
        calibration["sensor_pose"].inv().apply(traj_xyz),
        dtype=np.float64,
    )


def _project_camera_points(
    camera_xyz: np.ndarray,
    calibration: dict[str, Any],
) -> np.ndarray:
    """Project camera-frame points into image coordinates."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(
            calibration["camera_model"].ray2pixel(camera_xyz),
            dtype=np.float64,
        )


def _near_plane_intersection(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Intersect a camera-space edge with the positive-z near plane."""
    fraction = (CAMERA_NEAR_PLANE_M - start[2]) / (end[2] - start[2])
    intersection = start + fraction * (end - start)
    intersection[2] = CAMERA_NEAR_PLANE_M
    return intersection


def _clip_segment_to_near_plane(
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray | None:
    """Clip a camera-space segment to the positive-z near plane."""
    start_visible = start[2] >= CAMERA_NEAR_PLANE_M
    end_visible = end[2] >= CAMERA_NEAR_PLANE_M
    if not start_visible and not end_visible:
        return None
    if start_visible and end_visible:
        return np.asarray([start, end])

    intersection = _near_plane_intersection(start, end)
    if start_visible:
        return np.asarray([start, intersection])
    return np.asarray([intersection, end])


def _clip_polygon_to_near_plane(vertices: np.ndarray) -> np.ndarray:
    """Clip a camera-space polygon to the positive-z near plane."""
    if not len(vertices):
        return np.empty((0, 3), dtype=np.float64)

    clipped = []
    previous = vertices[-1]
    previous_visible = previous[2] >= CAMERA_NEAR_PLANE_M
    for current in vertices:
        current_visible = current[2] >= CAMERA_NEAR_PLANE_M
        if current_visible:
            if not previous_visible:
                clipped.append(_near_plane_intersection(previous, current))
            clipped.append(current)
        elif previous_visible:
            clipped.append(_near_plane_intersection(previous, current))
        previous = current
        previous_visible = current_visible
    return np.asarray(clipped, dtype=np.float64).reshape(-1, 3)


def _image_bounds(image_shape: tuple[int, int]) -> tuple[float, float, float, float]:
    """Return Matplotlib's half-pixel image bounds."""
    height, width = image_shape
    return -0.5, width - 0.5, -0.5, height - 0.5


def _clip_line_to_image(
    start: np.ndarray,
    end: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray | None:
    """Clip a projected line segment to the image rectangle."""
    if not np.isfinite([start, end]).all():
        return None

    x_min, x_max, y_min, y_max = _image_bounds(image_shape)
    delta = end - start
    lower_fraction = 0.0
    upper_fraction = 1.0
    for coefficient, distance in (
        (-delta[0], start[0] - x_min),
        (delta[0], x_max - start[0]),
        (-delta[1], start[1] - y_min),
        (delta[1], y_max - start[1]),
    ):
        if np.isclose(coefficient, 0.0):
            if distance < 0:
                return None
            continue
        fraction = distance / coefficient
        if coefficient < 0:
            if fraction > upper_fraction:
                return None
            lower_fraction = max(lower_fraction, fraction)
        else:
            if fraction < lower_fraction:
                return None
            upper_fraction = min(upper_fraction, fraction)

    return np.asarray(
        [
            start + lower_fraction * delta,
            start + upper_fraction * delta,
        ]
    )


def _clip_polygon_half_space(
    vertices: np.ndarray,
    axis: int,
    boundary: float,
    keep_greater: bool,
) -> np.ndarray:
    """Clip a projected polygon to one axis-aligned half-space."""
    if not len(vertices):
        return vertices

    def is_inside(vertex: np.ndarray) -> bool:
        if keep_greater:
            return bool(vertex[axis] >= boundary)
        return bool(vertex[axis] <= boundary)

    clipped = []
    previous = vertices[-1]
    previous_inside = is_inside(previous)
    for current in vertices:
        current_inside = is_inside(current)
        if current_inside != previous_inside:
            fraction = (boundary - previous[axis]) / (current[axis] - previous[axis])
            intersection = previous + fraction * (current - previous)
            intersection[axis] = boundary
            clipped.append(intersection)
        if current_inside:
            clipped.append(current)
        previous = current
        previous_inside = current_inside
    return np.asarray(clipped, dtype=np.float64).reshape(-1, 2)


def _polygon_area(vertices: np.ndarray) -> float:
    """Return the unsigned area of a projected polygon."""
    if len(vertices) < 3:
        return 0.0
    return float(
        0.5
        * abs(
            np.dot(vertices[:, 0], np.roll(vertices[:, 1], -1))
            - np.dot(vertices[:, 1], np.roll(vertices[:, 0], -1))
        )
    )


def _clip_polygon_to_image(
    vertices: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Clip a projected polygon to the image rectangle."""
    if not len(vertices) or not np.isfinite(vertices).all():
        return np.empty((0, 2), dtype=np.float64)

    original_area = _polygon_area(vertices)
    x_min, x_max, y_min, y_max = _image_bounds(image_shape)
    clipped = vertices
    for axis, boundary, keep_greater in (
        (0, x_min, True),
        (0, x_max, False),
        (1, y_min, True),
        (1, y_max, False),
    ):
        clipped = _clip_polygon_half_space(clipped, axis, boundary, keep_greater)
        if not len(clipped):
            return clipped

    if original_area > 1e-9 and _polygon_area(clipped) <= 1e-9:
        return np.empty((0, 2), dtype=np.float64)
    return clipped


def _trajectory_visualization_metadata() -> dict[str, Any]:
    """Describe the shared camera-overlay and BEV trajectory styling."""
    return {
        "camera_ribbon_width_m": TRAJECTORY_RIBBON_WIDTH_M,
        "camera_ribbon_semantics": "approximate vehicle-width corridor",
        "prediction_layer": "above_ground_truth",
        "trajectory_palette": {
            "prediction": TRAJECTORY_PREDICTION_COLOR,
            "ground_truth": TRAJECTORY_GROUND_TRUTH_COLOR,
            "history": TRAJECTORY_HISTORY_COLOR,
            "ego": TRAJECTORY_EGO_COLOR,
        },
    }


def _trajectory_ribbon_edges(
    traj_xyz: np.ndarray,
    width_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Offset trajectory edges by a fixed ground-plane width in meters."""
    point_count = len(traj_xyz)
    if point_count == 0:
        return traj_xyz.copy(), traj_xyz.copy()
    if point_count == 1:
        tangent_xy = np.array([[1.0, 0.0]], dtype=traj_xyz.dtype)
    else:
        tangent_xy = np.gradient(traj_xyz[:, :2], axis=0)

    tangent_norm = np.linalg.norm(tangent_xy, axis=1, keepdims=True)
    moving = tangent_norm[:, 0] > 1e-6
    if moving.any():
        moving_indices = np.flatnonzero(moving)
        nearest_moving = np.abs(np.arange(point_count)[:, None] - moving_indices).argmin(axis=1)
        tangent_xy[~moving] = tangent_xy[moving_indices[nearest_moving[~moving]]]
    else:
        tangent_xy[:] = (1.0, 0.0)
    tangent_norm = np.linalg.norm(tangent_xy, axis=1, keepdims=True)
    tangent_xy = tangent_xy / np.maximum(tangent_norm, 1e-6)
    normal_xy = np.column_stack((-tangent_xy[:, 1], tangent_xy[:, 0]))
    offset_xyz = np.zeros_like(traj_xyz)
    offset_xyz[:, :2] = normal_xy * (width_m / 2.0)
    return traj_xyz + offset_xyz, traj_xyz - offset_xyz


def _merge_connected_segments(segments: list[np.ndarray]) -> list[np.ndarray]:
    """Merge adjacent clipped line segments into plot-ready runs."""
    runs: list[list[np.ndarray]] = []
    for segment in segments:
        if runs and np.allclose(runs[-1][-1], segment[0]):
            runs[-1].append(segment[1])
        else:
            runs.append([segment[0], segment[1]])
    return [np.asarray(run) for run in runs]


def _draw_trajectory_ribbon(
    ax: plt.Axes,
    traj_xyz: np.ndarray,
    calibration: dict[str, Any],
    image_shape: tuple[int, int],
    color: str,
    dashed: bool,
    fill_alpha: float,
    zorder: int,
) -> bool:
    """Project a ground-plane trajectory ribbon and its centerline."""
    left_xyz, right_xyz = _trajectory_ribbon_edges(traj_xyz, TRAJECTORY_RIBBON_WIDTH_M)
    left_camera = _camera_coordinates(left_xyz, calibration)
    right_camera = _camera_coordinates(right_xyz, calibration)
    center_camera = _camera_coordinates(traj_xyz, calibration)

    quads = []
    for index in range(len(traj_xyz) - 1):
        camera_quad = _clip_polygon_to_near_plane(
            np.asarray(
                [
                    left_camera[index],
                    left_camera[index + 1],
                    right_camera[index + 1],
                    right_camera[index],
                ]
            )
        )
        if not len(camera_quad):
            continue
        projected_quad = _project_camera_points(camera_quad, calibration)
        clipped_quad = _clip_polygon_to_image(projected_quad, image_shape)
        if len(clipped_quad):
            quads.append(clipped_quad)
    if quads:
        ax.add_collection(
            PolyCollection(
                quads,
                facecolors=color,
                edgecolors="none",
                alpha=fill_alpha,
                zorder=zorder,
            )
        )

    projected_any = bool(quads)
    projected_segments = []
    for index in range(len(traj_xyz) - 1):
        camera_segment = _clip_segment_to_near_plane(
            center_camera[index],
            center_camera[index + 1],
        )
        if camera_segment is None:
            continue
        projected_segment = _project_camera_points(camera_segment, calibration)
        clipped_segment = _clip_line_to_image(
            projected_segment[0],
            projected_segment[1],
            image_shape,
        )
        if clipped_segment is not None:
            projected_segments.append(clipped_segment)

    runs = _merge_connected_segments(projected_segments)
    for run in runs:
        ax.plot(
            run[:, 0],
            run[:, 1],
            color=color,
            linewidth=3.0,
            linestyle="--" if dashed else "-",
            dash_capstyle="round",
            solid_capstyle="round",
            zorder=zorder + 1,
        )
        projected_any = True
    return projected_any


def _plot_camera_panel(
    ax: plt.Axes,
    image: np.ndarray,
    pred_xyz: np.ndarray,
    gt_xyz: np.ndarray | None,
    calibration: dict[str, Any] | None,
    title: str,
) -> bool:
    """Plot one camera image and optional projected trajectories."""
    ax.imshow(image)
    ax.set_xlim(-0.5, image.shape[1] - 0.5)
    ax.set_ylim(image.shape[0] - 0.5, -0.5)
    ax.set_autoscale_on(False)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    if calibration is None:
        return False

    projected_any = False
    if gt_xyz is not None:
        projected_any |= _draw_trajectory_ribbon(
            ax,
            gt_xyz,
            calibration,
            image.shape[:2],
            TRAJECTORY_GROUND_TRUTH_COLOR,
            dashed=True,
            fill_alpha=0.10,
            zorder=3,
        )
    projected_any |= _draw_trajectory_ribbon(
        ax,
        np.median(pred_xyz, axis=0),
        calibration,
        image.shape[:2],
        TRAJECTORY_PREDICTION_COLOR,
        dashed=False,
        fill_alpha=0.18,
        zorder=5,
    )
    return projected_any


def _plot_image_panel(
    ax: plt.Axes,
    image: np.ndarray,
    title: str,
    grounding_boxes: list[dict[str, Any]] | None = None,
    grounding_coordinate_mode: str = "auto",
) -> None:
    """Plot one image panel in the public blog style."""
    ax.imshow(image)
    if grounding_boxes:
        _draw_grounding_boxes(
            ax,
            image.shape[:2],
            grounding_boxes,
            coordinate_mode=grounding_coordinate_mode,
        )
    ax.set_title(title, fontsize=BLOG_STYLE["camera_title_fontsize"], pad=7)
    ax.axis("off")


def _parse_grounding_boxes(raw: str | None) -> list[dict[str, Any]]:
    """Parse Qwen/legacy grounding box text into normalized metadata entries."""
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []

    parsed: Any | None = None
    if raw[0] in "[{":
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                if raw.startswith("["):
                    try:
                        parsed = ast.literal_eval(f"[{raw}]")
                    except (ValueError, SyntaxError):
                        parsed = None
    if parsed is None:
        return [
            {
                "bbox": [float(match.group(idx)) for idx in range(1, 5)],
                "label": match.group(5) or "",
            }
            for match in GROUNDING_BBOX_FALLBACK_RE.finditer(raw)
        ]
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []

    boxes: list[dict[str, Any]] = []
    for item in parsed:
        label = None
        bbox = None
        if isinstance(item, dict):
            label = item.get("label")
            for key in ("bbox_2d", "bbox", "box"):
                if key in item:
                    bbox = item[key]
                    break
        elif isinstance(item, (list, tuple)):
            bbox = item
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            coords = [float(value) for value in bbox]
        except (TypeError, ValueError):
            continue
        boxes.append({"bbox": coords, "label": "" if label is None else str(label)})
    return boxes


def _scale_grounding_box(
    bbox: list[float],
    image_shape: tuple[int, int],
    coordinate_mode: str = "auto",
) -> tuple[tuple[float, float, float, float], str]:
    """Scale a parsed grounding box to image pixels."""
    height, width = image_shape
    x1, y1, x2, y2 = bbox
    max_x = max(abs(x1), abs(x2))
    max_y = max(abs(y1), abs(y2))
    max_value = max(max_x, max_y)

    if coordinate_mode == "normalized" or (coordinate_mode == "auto" and max_value <= 1.5):
        mode = "normalized"
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    elif coordinate_mode == "qwen_1000" or (
        coordinate_mode == "auto" and max_value <= 1000 and (width > 1000 or height > 1000)
    ):
        mode = "qwen_1000"
        x1, x2 = x1 / 1000.0 * width, x2 / 1000.0 * width
        y1, y2 = y1 / 1000.0 * height, y2 / 1000.0 * height
    else:
        mode = "pixel"

    left, right = sorted((max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))))
    top, bottom = sorted((max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))))
    return (left, top, right, bottom), mode


def _draw_grounding_boxes(
    ax: plt.Axes,
    image_shape: tuple[int, int],
    grounding_boxes: list[dict[str, Any]],
    coordinate_mode: str = "auto",
) -> None:
    """Draw grounding boxes on an image axis."""
    for box in grounding_boxes:
        (left, top, right, bottom), _mode = _scale_grounding_box(
            box["bbox"],
            image_shape,
            coordinate_mode=coordinate_mode,
        )
        if right <= left or bottom <= top:
            continue
        rect = Rectangle(
            (left, top),
            right - left,
            bottom - top,
            linewidth=3.0,
            edgecolor="#00AEEF",
            facecolor="none",
        )
        ax.add_patch(rect)
        label = box.get("label")
        if label:
            ax.text(
                left,
                max(0.0, top - 8),
                str(label),
                ha="left",
                va="bottom",
                color="white",
                fontsize=13,
                fontweight="semibold",
                bbox={"facecolor": "#00AEEF", "edgecolor": "none", "pad": 3, "alpha": 0.9},
            )


def _make_filmstrip(frames: torch.Tensor) -> np.ndarray:
    """Convert ``[T, 3, H, W]`` frames to one compact filmstrip image."""
    images = [_tensor_to_image(frame) for frame in frames]
    if len(images) == 1:
        return images[0]
    if len(images) == 4:
        height = images[0].shape[0]
        width = images[0].shape[1]
        h_separator = np.full((height, 4, 3), 255, dtype=np.uint8)
        v_separator = np.full((4, width * 2 + 4, 3), 255, dtype=np.uint8)
        top = np.concatenate([images[0], h_separator, images[1]], axis=1)
        bottom = np.concatenate([images[2], h_separator, images[3]], axis=1)
        return np.concatenate([top, v_separator, bottom], axis=0)

    height = images[0].shape[0]
    separator = np.full((height, 4, 3), 255, dtype=np.uint8)
    strip_parts: list[np.ndarray] = []
    for frame_idx, image in enumerate(images):
        if frame_idx:
            strip_parts.append(separator)
        strip_parts.append(image)
    return np.concatenate(strip_parts, axis=1)


def _task_camera_grid_layout(camera_indices: list[int]) -> dict[int, tuple[int, int]]:
    """Return the aligned two-row layout for a validated public task profile."""
    camera_ids = tuple(int(camera_id) for camera_id in camera_indices)
    if camera_ids == DRIVING_SIX_CAMERA_FOUR_FRAME.camera_ids:
        return DRIVING_CAMERA_GRID_LAYOUT
    if camera_ids == VQA_SIX_CAMERA_FOUR_FRAME.camera_ids:
        return VQA_CAMERA_GRID_LAYOUT
    raise ValueError(
        f"visualization requires a validated six-camera task profile; got camera IDs {camera_ids}"
    )


def _plot_blog_camera_grid(
    fig: plt.Figure,
    grid: Any,
    data: dict[str, Any],
    use_filmstrip: bool = False,
    frame_index: int | None = None,
    grounding_boxes_by_camera: dict[int, list[dict[str, Any]]] | None = None,
    grounding_coordinate_mode: str = "auto",
) -> None:
    """Render the selected task profile as an aligned two-by-three camera grid."""
    image_frames = data["image_frames"]
    camera_indices = data["camera_indices"].tolist()
    camera_layout = _task_camera_grid_layout(camera_indices)

    for camera_idx, position in camera_layout.items():
        ax = fig.add_subplot(grid[position])
        camera_title = CAMERA_TITLES.get(camera_idx, f"camera {camera_idx}")
        source_idx = camera_indices.index(camera_idx)
        if use_filmstrip:
            image = _make_filmstrip(image_frames[source_idx])
        elif frame_index is not None:
            image = _tensor_to_image(image_frames[source_idx, frame_index])
        else:
            image = _tensor_to_image(image_frames[source_idx, -1])
        _plot_image_panel(
            ax,
            image,
            camera_title,
            grounding_boxes=(grounding_boxes_by_camera or {}).get(camera_idx),
            grounding_coordinate_mode=grounding_coordinate_mode,
        )


def _style_text_axis(ax: plt.Axes) -> None:
    """Style one full-width text panel."""
    ax.set_facecolor("#fbfbfb")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#c8c8c8")
        spine.set_linewidth(1.2)


def _add_text_panel(
    ax: plt.Axes,
    title: str,
    body: str,
    title_fontsize: int | None = None,
    body_fontsize: int | None = None,
    min_body_fontsize: int = 12,
    single_line: bool = False,
) -> _TextPanel:
    """Draw a full-width blog-style text panel."""
    _style_text_axis(ax)
    ax.text(
        0.025,
        0.82,
        title,
        ha="left",
        va="top",
        fontsize=title_fontsize or BLOG_STYLE["task_title_fontsize"],
        fontweight="semibold",
        transform=ax.transAxes,
    )
    body = body or "(not provided)"
    body_artist = ax.text(
        0.025,
        0.34,
        body,
        ha="left",
        va="center",
        fontsize=body_fontsize or BLOG_STYLE["task_body_fontsize"],
        linespacing=1.10,
        clip_on=True,
        transform=ax.transAxes,
    )
    return _TextPanel(
        ax=ax,
        body_artist=body_artist,
        body=body,
        max_fontsize=body_fontsize or BLOG_STYLE["task_body_fontsize"],
        min_fontsize=min_body_fontsize,
        single_line=single_line,
    )


def _wrap_text_to_width(
    renderer: Any,
    font_properties: FontProperties,
    body: str,
    max_width_px: float,
) -> str:
    """Wrap text using measured glyph widths instead of character counts."""
    wrapped_lines: list[str] = []
    for paragraph in body.splitlines() or [body]:
        words = paragraph.split()
        if not words:
            wrapped_lines.append("")
            continue
        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            width, _, _ = renderer.get_text_width_height_descent(
                candidate,
                font_properties,
                ismath=False,
            )
            if width <= max_width_px:
                line = candidate
            else:
                wrapped_lines.append(line)
                line = word
        wrapped_lines.append(line)
    return "\n".join(wrapped_lines)


def _fit_text_panels(panels: list[_TextPanel]) -> None:
    """Fit panel bodies inside their borders while preserving readable type."""
    if not panels:
        return
    fig = panels[0].ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for panel in panels:
        region = Bbox.from_extents(0.025, 0.06, 0.975, 0.63).transformed(panel.ax.transAxes)
        normalized_body = " ".join(panel.body.split()) if panel.single_line else panel.body.strip()
        for fontsize in range(panel.max_fontsize, panel.min_fontsize - 1, -1):
            font_properties = panel.body_artist.get_fontproperties().copy()
            font_properties.set_size(fontsize)
            rendered_body = (
                normalized_body
                if panel.single_line
                else _wrap_text_to_width(
                    renderer,
                    font_properties,
                    normalized_body,
                    region.width,
                )
            )
            panel.body_artist.set_fontsize(fontsize)
            panel.body_artist.set_text(rendered_body)
            panel.body_artist.set_position((0.025, 0.345))
            text_bounds = panel.body_artist.get_window_extent(renderer)
            if text_bounds.width <= region.width and text_bounds.height <= region.height:
                break
    fig.canvas.draw()


def _base_blog_metadata(
    data: dict[str, Any],
    task: str,
    figure_style: str,
    model_id: str | None,
    seed: int | None,
) -> dict[str, Any]:
    """Build JSON-friendly metadata shared by public blog-style figures."""
    camera_grid_camera_ids = list(_task_camera_grid_layout(data["camera_indices"].tolist()))
    return {
        "task": task,
        "figure_style": figure_style,
        "clip_id": data.get("clip_id"),
        "t0_us": _json_scalar(data.get("t0_us")),
        "camera_tmin": _json_scalar(data.get("camera_tmin")),
        "camera_indices": data["camera_indices"].tolist(),
        "input_profile": data.get("input_profile"),
        "camera_grid_camera_ids": camera_grid_camera_ids,
        "camera_titles": [CAMERA_TITLES.get(camera_idx) for camera_idx in camera_grid_camera_ids],
        "model_id": model_id,
        **_checkpoint_metadata(model_id),
        "seed": seed,
        "image_frames_shape": _shape(data.get("image_frames")),
        "absolute_timestamps_shape": _shape(data.get("absolute_timestamps")),
        "relative_timestamps_shape": _shape(data.get("relative_timestamps")),
    }


def _save_figure_and_metadata(
    fig: plt.Figure,
    metadata: dict[str, Any],
    output_path: str | Path | None,
    json_path: str | Path | None,
    dpi: int = 180,
) -> None:
    """Optionally save figure and JSON metadata."""
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
    if json_path is not None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _parse_meta_action_fields(meta_action: str) -> dict[str, str]:
    """Split no-special meta-action text into longitudinal/lateral/lane fields."""
    fields: dict[str, str] = {}
    for line in meta_action.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in META_ACTION_FIELDS:
            fields[key] = value.strip()
    return fields


def _format_meta_action_fields(meta_action_fields: dict[str, str], meta_action: str) -> str:
    """Format parsed meta-action fields for display."""
    if not meta_action_fields:
        return meta_action
    return "\n".join(
        f"{field}: {meta_action_fields.get(field, '')}" for field in META_ACTION_FIELDS
    )


def _format_grounding_box_result(
    grounding_boxes: list[dict[str, Any]],
    grounding_camera_ids: tuple[int, ...],
) -> str:
    """Format labels and coordinates for the visible grounding result panel."""
    camera_names = [
        CAMERA_TITLES.get(camera_id, f"camera {camera_id}") for camera_id in grounding_camera_ids
    ]
    camera_text = ", ".join(camera_names)
    lines = []
    for box in grounding_boxes:
        coordinates = ", ".join(f"{float(value):g}" for value in box["bbox"])
        fields = [field for field in (camera_text, box.get("label")) if field]
        fields.append(f"bbox_2d: [{coordinates}]")
        lines.append(" | ".join(str(field) for field in fields))
    return "\n".join(lines)


def _format_auto_labeling_json(auto_labeling_json: dict[str, Any]) -> str:
    """Format the fixed auto-labeling schema as readable text."""
    labels = {
        "critical_components_analysis": "Critical Components",
        "ego_vehicle_motion_analysis": "Ego Motion",
        "trajectory_analysis": "Trajectory",
        "chain_of_causation": "Chain of Causation",
    }
    lines = []
    for key in AUTO_LABELING_KEYS:
        value = auto_labeling_json.get(key)
        if value is None:
            value = ""
        lines.append(f"{labels[key]}: {value}")
    return "\n\n".join(lines)


def _add_auto_label_field_panel(
    ax: plt.Axes,
    title: str,
    value: Any,
) -> _TextPanel:
    """Draw one readable auto-labeling field panel."""
    body = "" if value is None else str(value)
    return _add_text_panel(
        ax,
        title,
        body,
        title_fontsize=23,
        body_fontsize=18,
        min_body_fontsize=12,
    )


def _plot_bev(
    ax: plt.Axes,
    pred_xyz: np.ndarray,
    gt_xyz: np.ndarray | None,
    history_xyz: np.ndarray | None,
) -> None:
    """Plot trajectories in ego-frame BEV."""
    for sample_idx in range(pred_xyz.shape[0]):
        ax.plot(
            pred_xyz[sample_idx, :, 0],
            pred_xyz[sample_idx, :, 1],
            color=TRAJECTORY_PREDICTION_COLOR,
            linewidth=0.9,
            alpha=0.25,
            zorder=2,
        )
    median_pred = np.median(pred_xyz, axis=0)
    ax.plot(
        median_pred[:, 0],
        median_pred[:, 1],
        color=TRAJECTORY_PREDICTION_COLOR,
        linewidth=2.5,
        label="pred",
        zorder=5,
    )
    if gt_xyz is not None:
        ax.plot(
            gt_xyz[:, 0],
            gt_xyz[:, 1],
            color=TRAJECTORY_GROUND_TRUTH_COLOR,
            linewidth=2.5,
            linestyle="--",
            label="gt",
            zorder=4,
        )
    if history_xyz is not None:
        ax.plot(
            history_xyz[:, 0],
            history_xyz[:, 1],
            color=TRAJECTORY_HISTORY_COLOR,
            linewidth=2.0,
            linestyle="-.",
            label="history",
            zorder=3,
        )
    ax.scatter([0], [0], color=TRAJECTORY_EGO_COLOR, s=24, label="ego", zorder=6)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    _enforce_bev_axes(ax)


def _style_blog_bev(ax: plt.Axes) -> None:
    """Format the BEV plot for the public task-native blog figure."""
    old_legend = ax.get_legend()
    if old_legend is not None:
        old_legend.remove()

    legend_labels = {
        "pred": "Predicted Trajectory",
        "gt": "Ground Truth Motion",
        "history": "Ego History",
        "ego": "Ego Vehicle",
    }
    ax.set_title(
        "Predicted Trajectory vs Ground Truth Motion",
        fontsize=BLOG_STYLE["bev_title_fontsize"],
        pad=12,
        fontweight="semibold",
    )
    ax.set_xlabel("x (m)", fontsize=18)
    ax.set_ylabel("y (m)", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    for line in ax.lines:
        alpha = line.get_alpha()
        line.set_linewidth(1.9 if alpha is not None and alpha < 0.5 else 4.2)
    for collection in ax.collections:
        collection.set_sizes([68])

    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    target_ratio = 4.1
    required_xrange = max(xhi - xlo, (yhi - ylo) * target_ratio)
    xmid = (xlo + xhi) / 2
    ax.set_xlim(xmid - required_xrange / 2, xmid + required_xrange / 2)

    handles, labels = ax.get_legend_handles_labels()
    labels = [legend_labels.get(label, label) for label in labels]
    ax.legend(
        handles,
        labels,
        loc="upper center",
        ncols=4,
        fontsize=BLOG_STYLE["bev_legend_fontsize"],
        frameon=True,
        borderpad=0.35,
        borderaxespad=0.35,
        labelspacing=0.20,
        handlelength=2.0,
        markerscale=1.25,
    )


def _enforce_bev_axes(ax: plt.Axes) -> None:
    """Keep low-lateral-motion BEV scenes readable."""
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    xrange = max(xhi - xlo, MIN_BEV_AXIS_RANGE_M)
    yrange = max(yhi - ylo, MIN_BEV_AXIS_RANGE_M, MIN_BEV_LATERAL_RATIO * xrange)

    xmid = (xlo + xhi) / 2
    ymid = (ylo + yhi) / 2
    ax.set_xlim(xmid - xrange / 2, xmid + xrange / 2)
    ax.set_ylim(ymid - yrange / 2, ymid + yrange / 2)


def plot_compact_inference_result(
    data: dict[str, Any],
    pred_xyz: torch.Tensor,
    extra: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
    json_path: str | Path | None = None,
    model_id: str | None = None,
    seed: int | None = None,
    require_camera_projection: bool = False,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Plot compact camera views, projected trajectories, BEV, and generated CoT.

    Args:
        data: Sample dictionary returned by ``load_physical_aiavdataset``.
        pred_xyz: Model trajectories, shape ``[B, num_sets, K, T, 3]``.
        extra: Optional text output dict from ``sample_trajectories_from_data``.
        output_path: Optional PNG path to save.
        json_path: Optional JSON sidecar path to save.
        model_id: Model id/path recorded in the JSON sidecar.
        seed: Random seed recorded in the JSON sidecar.
        require_camera_projection: Raise if no camera overlay can be projected.

    Returns:
        ``(figure, metadata)``.
    """
    pred_np = pred_xyz.detach().cpu().numpy()[0, 0]
    gt_future = data.get("ego_future_xyz")
    gt_np = None if gt_future is None else gt_future.detach().cpu().numpy()[0, 0]
    history_np = data["ego_history_xyz"].detach().cpu().numpy()[0, 0]
    metrics = _compute_metrics(pred_xyz, gt_future)
    cots = _extract_cots(extra)
    cot = cots[0] if cots else ""

    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.25])
    image_frames = data["image_frames"]
    camera_indices = data["camera_indices"].tolist()
    camera_calibrations = data.get("camera_calibrations", {})
    projected_cameras: list[int] = []

    for camera_idx, (row, col) in CAMERA_GRID_LAYOUT.items():
        ax = fig.add_subplot(grid[row, col])
        if camera_idx not in camera_indices:
            ax.axis("off")
            continue
        source_idx = camera_indices.index(camera_idx)
        image = _tensor_to_image(image_frames[source_idx, -1])
        projected = _plot_camera_panel(
            ax=ax,
            image=image,
            pred_xyz=pred_np,
            gt_xyz=gt_np,
            calibration=camera_calibrations.get(camera_idx),
            title=CAMERA_TITLES.get(camera_idx, f"camera {camera_idx}"),
        )
        if projected:
            projected_cameras.append(camera_idx)

    bev_ax = fig.add_subplot(grid[2, :2])
    _plot_bev(bev_ax, pred_np, gt_np, history_np)

    text_ax = fig.add_subplot(grid[2, 2])
    text_ax.axis("off")
    text_ax.text(
        0,
        1,
        cot or "(no CoT decoded)",
        ha="left",
        va="top",
        wrap=True,
        fontsize=9,
    )
    text_ax.set_title("Predicted CoT", fontsize=10)

    projection_available = bool(projected_cameras)
    if require_camera_projection and not projection_available:
        raise RuntimeError("No camera trajectory projection was available for this sample.")

    metadata = {
        "figure_style": "compact",
        "clip_id": data.get("clip_id"),
        "t0_us": _json_scalar(data.get("t0_us")),
        "camera_tmin": _json_scalar(data.get("camera_tmin")),
        "camera_indices": camera_indices,
        "input_profile": data.get("input_profile"),
        "model_id": model_id,
        **_checkpoint_metadata(model_id),
        "seed": seed,
        "cot": cot,
        "cots": cots,
        "projection_available": projection_available,
        "projected_camera_ids": projected_cameras,
        **_trajectory_visualization_metadata(),
        "num_trajectory_samples": int(pred_np.shape[0]),
        "pred_xyz_shape": _shape(pred_xyz),
        "image_frames_shape": _shape(data.get("image_frames")),
        "absolute_timestamps_shape": _shape(data.get("absolute_timestamps")),
        "relative_timestamps_shape": _shape(data.get("relative_timestamps")),
        "ego_history_xyz_shape": _shape(data.get("ego_history_xyz")),
        "ego_future_xyz_shape": _shape(gt_future),
        **metrics,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
    if json_path is not None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return fig, metadata


def plot_inference_result(
    data: dict[str, Any],
    pred_xyz: torch.Tensor,
    extra: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
    json_path: str | Path | None = None,
    model_id: str | None = None,
    seed: int | None = None,
    require_camera_projection: bool = False,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Plot the public task-native six-camera figure for one inference result.

    Args:
        data: Six-camera sample returned by ``select_task_input``.
        pred_xyz: Model trajectories, shape ``[B, num_sets, K, T, 3]``.
        extra: Optional text output dict from ``sample_trajectories_from_data``.
        output_path: Optional PNG path to save.
        json_path: Optional JSON sidecar path to save.
        model_id: Model id/path recorded in the JSON sidecar.
        seed: Random seed recorded in the JSON sidecar.
        require_camera_projection: Raise if no camera overlay can be projected.

    Returns:
        ``(figure, metadata)``.
    """
    pred_np = pred_xyz.detach().cpu().numpy()[0, 0]
    gt_future = data.get("ego_future_xyz")
    gt_np = None if gt_future is None else gt_future.detach().cpu().numpy()[0, 0]
    history_np = data["ego_history_xyz"].detach().cpu().numpy()[0, 0]
    metrics = _compute_metrics(pred_xyz, gt_future)
    cots = _extract_cots(extra)
    cot = cots[0] if cots else ""

    fig = plt.figure(figsize=(16.5, 13.0), constrained_layout=True)
    grid = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.0, 1.18, 0.78])
    image_frames = data["image_frames"]
    camera_indices = data["camera_indices"].tolist()
    camera_calibrations = data.get("camera_calibrations", {})
    projected_cameras: list[int] = []
    camera_layout = _task_camera_grid_layout(camera_indices)

    for camera_idx, position in camera_layout.items():
        ax = fig.add_subplot(grid[position])
        camera_title = CAMERA_TITLES.get(camera_idx, f"camera {camera_idx}")
        source_idx = camera_indices.index(camera_idx)
        image = _tensor_to_image(image_frames[source_idx, -1])
        projected = _plot_camera_panel(
            ax=ax,
            image=image,
            pred_xyz=pred_np,
            gt_xyz=gt_np,
            calibration=camera_calibrations.get(camera_idx),
            title=camera_title,
        )
        ax.set_title(camera_title, fontsize=BLOG_STYLE["camera_title_fontsize"], pad=7)
        for line in ax.lines:
            line.set_linewidth(max(line.get_linewidth() * 1.35, 1.5))
        if projected:
            projected_cameras.append(camera_idx)

    bev_ax = fig.add_subplot(grid[2, :])
    _plot_bev(bev_ax, pred_np, gt_np, history_np)
    _style_blog_bev(bev_ax)

    cot_panel = _add_text_panel(
        fig.add_subplot(grid[3, :]),
        BLOG_COC_LABEL,
        cot or "(no CoC decoded)",
        title_fontsize=BLOG_STYLE["coc_title_fontsize"],
        body_fontsize=BLOG_STYLE["coc_body_fontsize"],
        single_line=True,
    )
    _fit_text_panels([cot_panel])

    projection_available = bool(projected_cameras)
    if require_camera_projection and not projection_available:
        raise RuntimeError("No camera trajectory projection was available for this sample.")

    camera_grid_camera_ids = list(camera_layout)
    metadata = {
        "figure_style": "blog_6cam_task_native",
        "clip_id": data.get("clip_id"),
        "t0_us": _json_scalar(data.get("t0_us")),
        "camera_tmin": _json_scalar(data.get("camera_tmin")),
        "camera_indices": camera_indices,
        "input_profile": data.get("input_profile"),
        "camera_grid_camera_ids": camera_grid_camera_ids,
        "camera_titles": [CAMERA_TITLES.get(camera_idx) for camera_idx in camera_grid_camera_ids],
        "coc_label": BLOG_COC_LABEL,
        "model_id": model_id,
        **_checkpoint_metadata(model_id),
        "seed": seed,
        "cot": cot,
        "cots": cots,
        "projection_available": projection_available,
        "projected_camera_ids": projected_cameras,
        **_trajectory_visualization_metadata(),
        "num_trajectory_samples": int(pred_np.shape[0]),
        "pred_xyz_shape": _shape(pred_xyz),
        "image_frames_shape": _shape(data.get("image_frames")),
        "absolute_timestamps_shape": _shape(data.get("absolute_timestamps")),
        "relative_timestamps_shape": _shape(data.get("relative_timestamps")),
        "ego_history_xyz_shape": _shape(data.get("ego_history_xyz")),
        "ego_future_xyz_shape": _shape(gt_future),
        "bev_legend_fontsize": BLOG_STYLE["bev_legend_fontsize"],
        **metrics,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180)
    if json_path is not None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return fig, metadata


def plot_blog_figure(
    data: dict[str, Any],
    pred_xyz: torch.Tensor,
    extra: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
    json_path: str | Path | None = None,
    model_id: str | None = None,
    seed: int | None = None,
    require_camera_projection: bool = False,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Plot the public task-native six-camera figure for one inference result."""
    return plot_inference_result(
        data=data,
        pred_xyz=pred_xyz,
        extra=extra,
        output_path=output_path,
        json_path=json_path,
        model_id=model_id,
        seed=seed,
        require_camera_projection=require_camera_projection,
    )


def plot_meta_action_result(
    data: dict[str, Any],
    cot: str,
    meta_action: str,
    output_path: str | Path | None = None,
    json_path: str | Path | None = None,
    model_id: str | None = None,
    seed: int | None = None,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Plot task-profile camera views and generated meta-action text."""
    fig = plt.figure(figsize=(16.5, 10.2), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.78])
    _plot_blog_camera_grid(fig, grid, data)

    meta_action_fields = _parse_meta_action_fields(meta_action)
    meta_action_ax = fig.add_subplot(grid[2, :])
    panel = _add_text_panel(
        meta_action_ax,
        "Predicted Meta-Action",
        _format_meta_action_fields(meta_action_fields, meta_action),
    )
    _fit_text_panels([panel])

    metadata = {
        **_base_blog_metadata(
            data=data,
            task="meta_action",
            figure_style="blog_meta_action",
            model_id=model_id,
            seed=seed,
        ),
        "cot": cot,
        "meta_action": meta_action,
        "meta_action_fields": meta_action_fields,
    }
    _save_figure_and_metadata(fig, metadata, output_path, json_path)
    return fig, metadata


def plot_vqa_result(
    data: dict[str, Any],
    question: str,
    answer: str,
    output_path: str | Path | None = None,
    json_path: str | Path | None = None,
    model_id: str | None = None,
    seed: int | None = None,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Plot task-profile camera views and plain VQA question/answer text."""
    fig = plt.figure(figsize=(16.5, 13.2), constrained_layout=True)
    grid = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.0, 0.78, 0.92])
    _plot_blog_camera_grid(fig, grid, data)

    question_ax = fig.add_subplot(grid[2, :])
    question_panel = _add_text_panel(question_ax, "Question", question, single_line=True)

    answer_ax = fig.add_subplot(grid[3, :])
    answer_panel = _add_text_panel(answer_ax, "Answer", answer)
    _fit_text_panels([question_panel, answer_panel])

    metadata = {
        **_base_blog_metadata(
            data=data,
            task="vqa",
            figure_style="blog_vqa",
            model_id=model_id,
            seed=seed,
        ),
        "question": question,
        "answer": answer,
    }
    _save_figure_and_metadata(fig, metadata, output_path, json_path)
    return fig, metadata


def plot_grounding_result(
    data: dict[str, Any],
    question: str,
    answer: str,
    grounding_text: str | None = None,
    grounding_boxes: list[dict[str, Any]] | None = None,
    grounding_camera_ids: tuple[int, ...] | list[int] | None = None,
    grounding_coordinate_mode: str = "auto",
    output_path: str | Path | None = None,
    json_path: str | Path | None = None,
    model_id: str | None = None,
    seed: int | None = None,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Plot task-profile camera views with generated grounding box overlays."""
    grounding_text = answer if grounding_text is None else grounding_text
    grounding_boxes = grounding_boxes or _parse_grounding_boxes(grounding_text)
    if grounding_camera_ids is None:
        grounding_camera_ids = (1,) if grounding_boxes else ()
    grounding_camera_ids = tuple(int(camera_id) for camera_id in grounding_camera_ids)
    grounding_boxes_by_camera = {
        camera_id: grounding_boxes for camera_id in grounding_camera_ids if grounding_boxes
    }

    fig = plt.figure(figsize=(16.5, 13.2), constrained_layout=True)
    grid = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.0, 0.78, 0.92])
    _plot_blog_camera_grid(
        fig,
        grid,
        data,
        grounding_boxes_by_camera=grounding_boxes_by_camera,
        grounding_coordinate_mode=grounding_coordinate_mode,
    )

    question_ax = fig.add_subplot(grid[2, :])
    question_panel = _add_text_panel(
        question_ax,
        "Grounding Prompt",
        question,
        single_line=True,
    )

    result_ax = fig.add_subplot(grid[3, :])
    grounding_result = _format_grounding_box_result(grounding_boxes, grounding_camera_ids)
    result_panel = _add_text_panel(
        result_ax,
        "Grounding Result",
        grounding_result or grounding_text or answer,
        single_line=len(grounding_boxes) <= 1,
    )
    _fit_text_panels([question_panel, result_panel])

    metadata = {
        **_base_blog_metadata(
            data=data,
            task="grounding",
            figure_style="blog_grounding",
            model_id=model_id,
            seed=seed,
        ),
        "question": question,
        "answer": answer,
        "grounding_text": grounding_text,
        "grounding_boxes": grounding_boxes,
        "grounding_box_count": len(grounding_boxes),
        "grounding_camera_ids": list(grounding_camera_ids),
        "grounding_coordinate_mode": grounding_coordinate_mode,
    }
    _save_figure_and_metadata(fig, metadata, output_path, json_path)
    return fig, metadata


def plot_auto_labeling_result(
    data: dict[str, Any],
    auto_labeling_json: dict[str, Any],
    auto_labeling_text: str,
    future_source: str,
    trajectory_cot: str = "",
    output_path: str | Path | None = None,
    video_path: str | Path | None = None,
    video_fps: float = 2.0,
    json_path: str | Path | None = None,
    model_id: str | None = None,
    seed: int | None = None,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Plot an auto-labeling poster and optionally write its synchronized MP4."""
    fig = _build_auto_labeling_figure(
        data=data,
        auto_labeling_json=auto_labeling_json,
        use_filmstrip=True,
    )

    image_frames_shape = _shape(data.get("image_frames"))
    num_frames_per_camera = None if image_frames_shape is None else image_frames_shape[1]
    video_frame_shape = None
    if video_path is not None:
        video_frame_shape = _write_auto_labeling_video(
            data=data,
            auto_labeling_json=auto_labeling_json,
            video_path=video_path,
            video_fps=video_fps,
        )

    metadata = {
        **_base_blog_metadata(
            data=data,
            task="auto_labeling",
            figure_style="blog_auto_labeling_video",
            model_id=model_id,
            seed=seed,
        ),
        "input_mode": "multi_camera_context_video",
        "num_cameras": len(data["camera_indices"]),
        "num_frames_per_camera": num_frames_per_camera,
        "video_frame_count": num_frames_per_camera if video_path is not None else 0,
        "video_frame_shape": video_frame_shape,
        "video_fps": float(video_fps) if video_path is not None else None,
        "video_path": None if video_path is None else str(video_path),
        "poster_path": None if output_path is None else str(output_path),
        "future_source": future_source,
        "trajectory_cot": trajectory_cot,
        "auto_labeling_text": auto_labeling_text,
        "auto_labeling_json": auto_labeling_json,
    }
    _save_figure_and_metadata(fig, metadata, output_path, json_path)
    return fig, metadata


def _build_auto_labeling_figure(
    data: dict[str, Any],
    auto_labeling_json: dict[str, Any],
    use_filmstrip: bool = False,
    frame_index: int | None = None,
    dpi: int = 100,
) -> plt.Figure:
    """Build one poster or temporal frame for the auto-labeling visual."""
    fig = plt.figure(figsize=(16.5, 18.0), constrained_layout=True)
    fig.set_dpi(dpi)
    grid = fig.add_gridspec(
        6,
        3,
        height_ratios=[1.0, 1.0, 0.78, 1.65, 0.62, 0.62],
    )
    _plot_blog_camera_grid(
        fig,
        grid,
        data,
        use_filmstrip=use_filmstrip,
        frame_index=frame_index,
    )

    panels = [
        _add_auto_label_field_panel(
            fig.add_subplot(grid[2, :]),
            "Critical Components",
            auto_labeling_json.get("critical_components_analysis"),
        ),
        _add_auto_label_field_panel(
            fig.add_subplot(grid[3, :]),
            "Ego Vehicle Motion",
            auto_labeling_json.get("ego_vehicle_motion_analysis"),
        ),
        _add_auto_label_field_panel(
            fig.add_subplot(grid[4, :]),
            "Trajectory Analysis",
            auto_labeling_json.get("trajectory_analysis"),
        ),
        _add_auto_label_field_panel(
            fig.add_subplot(grid[5, :]),
            "Chain of Causation",
            auto_labeling_json.get("chain_of_causation"),
        ),
    ]
    _fit_text_panels(panels)
    return fig


def _write_auto_labeling_video(
    data: dict[str, Any],
    auto_labeling_json: dict[str, Any],
    video_path: str | Path,
    video_fps: float,
) -> list[int]:
    """Write one synchronized composite frame per model-input timestep."""
    if video_fps <= 0:
        raise ValueError(f"video_fps must be positive, got {video_fps}")
    image_frames = data["image_frames"]
    if image_frames.ndim != 5 or image_frames.shape[1] < 1:
        raise ValueError("image_frames must have shape [camera, frame, channel, height, width]")

    video_path = Path(video_path)
    if video_path.suffix.lower() != ".mp4":
        raise ValueError(f"video_path must end in .mp4, got {video_path}")
    video_path.parent.mkdir(parents=True, exist_ok=True)

    video_frames = []
    for frame_index in range(image_frames.shape[1]):
        frame_fig = _build_auto_labeling_figure(
            data=data,
            auto_labeling_json=auto_labeling_json,
            frame_index=frame_index,
            dpi=120,
        )
        frame_fig.canvas.draw()
        frame = np.asarray(frame_fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(frame_fig)
        video_frames.append(frame)

    media.write_video(
        video_path,
        video_frames,
        fps=video_fps,
        codec="h264",
        crf=18,
        encoded_format="yuv420p",
    )
    return list(video_frames[0].shape)
