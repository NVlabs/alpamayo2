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

"""Minimal end-to-end inference smoke script."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from alpamayo2_super.common.constants import PUBLIC_MODEL_ID


MODEL_ID = PUBLIC_MODEL_ID
CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US = 5_100_000
FIGURE_STYLES = ("blog", "compact")
MANIFEST_PATH = Path(__file__).resolve().parents[2] / "examples" / "validation_samples.json"
REQUIRED_LOCAL_CHECKPOINT_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
)


def _is_local_path_like(model_id: str) -> bool:
    """Return True when ``model_id`` is intended to be a local filesystem path."""
    return model_id.startswith(("/", "./", "../", "~"))


def resolve_project_path(path: str | Path, project_root: Path) -> Path:
    """Resolve a notebook path relative to its repository root."""
    resolved_path = Path(path).expanduser()
    if resolved_path.is_absolute():
        return resolved_path
    return project_root / resolved_path


def validate_model_id(model_id: str) -> None:
    """Validate a local checkpoint layout before expensive inference setup."""
    model_path = Path(model_id).expanduser()
    if not model_path.exists() and not _is_local_path_like(model_id):
        return

    if not model_path.exists():
        raise FileNotFoundError(
            f"Local checkpoint path does not exist: {model_path}. "
            f"Pass a Hugging Face repo id such as {PUBLIC_MODEL_ID}, or pass a "
            "release-native checkpoint directory."
        )
    if not model_path.is_dir():
        raise NotADirectoryError(f"Local checkpoint path is not a directory: {model_path}")

    missing_files = [
        filename
        for filename in REQUIRED_LOCAL_CHECKPOINT_FILES
        if not (model_path / filename).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Local checkpoint is missing required file(s): {', '.join(missing_files)}. "
            f"Checked: {model_path}"
        )
    if not any(model_path.glob("*.safetensors")):
        raise FileNotFoundError(f"Local checkpoint has no *.safetensors weight files: {model_path}")


def load_validation_sample(manifest_path: Path, sample_index: int) -> dict[str, Any]:
    """Load one validation sample manifest entry."""
    samples = json.loads(manifest_path.read_text(encoding="utf-8"))["samples"]
    sample = samples[sample_index]
    return {
        "clip_id": sample["clip_id"],
        "t0_us": int(sample["t0_us"]),
    }


def run_smoke(
    model_id: str = MODEL_ID,
    clip_id: str = CLIP_ID,
    t0_us: int = T0_US,
    save_viz: str | None = None,
    save_json: str | None = None,
    figure_style: str = "blog",
    diffusion_steps: int = 10,
    seed: int = 42,
    require_camera_projection: bool = False,
) -> None:
    """Load one PhysicalAI-AV clip and sample one trajectory."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    validate_model_id(model_id)

    import numpy as np
    import torch

    from alpamayo2_super import helper
    from alpamayo2_super.input_profiles import select_task_input
    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super
    from alpamayo2_super.visualization import (
        plot_compact_inference_result,
        plot_inference_result,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Alpamayo 2 Super inference smoke requires a CUDA GPU. "
            "Check that an NVIDIA GPU is visible with nvidia-smi and CUDA_VISIBLE_DEVICES."
        )

    print(f"Loading dataset for clip_id: {clip_id}...")
    source_data = load_physical_aiavdataset(
        clip_id,
        t0_us=t0_us,
    )
    data = select_task_input(source_data, "trajectory")
    print("Dataset loaded.")

    model = Alpamayo2Super.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda:0")
    model_inputs = helper.prepare_model_inputs(data, model.config, model.tokenizer)
    model_inputs = helper.to_device(model_inputs, "cuda")

    torch.cuda.manual_seed_all(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, logprob, extra = model.sample_trajectories_from_data(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            diffusion_kwargs={"inference_step": diffusion_steps},
            return_extra=True,
        )

    del pred_rot, logprob
    print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])

    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    print("minADE:", diff.min(), "meters")

    if save_viz or save_json:
        if figure_style == "blog":
            plot_figure = plot_inference_result
        elif figure_style == "compact":
            plot_figure = plot_compact_inference_result
        else:
            raise ValueError(
                f"Unknown figure style: {figure_style}. Expected one of: {', '.join(FIGURE_STYLES)}"
            )

        _, metadata = plot_figure(
            data=data,
            pred_xyz=pred_xyz,
            extra=extra,
            output_path=save_viz,
            json_path=save_json,
            model_id=model_id,
            seed=seed,
            require_camera_projection=require_camera_projection,
        )
        print("figure_style:", metadata.get("figure_style", figure_style))
        print("projection_available:", metadata["projection_available"])
        if save_viz:
            print("Saved visualization:", save_viz)
        if save_json:
            print("Saved metadata:", save_json)


def main() -> None:
    """Run the default smoke script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default=os.environ.get("ALPAMAYO2_SUPER_MODEL_ID", MODEL_ID),
        help="Hugging Face repo ID or local checkpoint path.",
    )
    parser.add_argument(
        "--clip-id",
        default=os.environ.get("ALPAMAYO2_SUPER_CLIP_ID", CLIP_ID),
        help="PhysicalAI-AV clip ID to load.",
    )
    parser.add_argument(
        "--t0-us",
        default=os.environ.get("ALPAMAYO2_SUPER_T0_US", str(T0_US)),
        help="Clip timestamp in microseconds.",
    )
    parser.add_argument(
        "--manifest",
        default=os.environ.get("ALPAMAYO2_SUPER_VALIDATION_MANIFEST", str(MANIFEST_PATH)),
        help="Optional validation_samples.json manifest. Overrides --clip-id/--t0-us.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=int(os.environ.get("ALPAMAYO2_SUPER_SAMPLE_INDEX", "0")),
        help="Sample index to load from --manifest.",
    )
    parser.add_argument("--save-viz", default=None, help="Optional PNG output path.")
    parser.add_argument("--save-json", default=None, help="Optional JSON sidecar output path.")
    parser.add_argument(
        "--figure-style",
        choices=FIGURE_STYLES,
        default=os.environ.get("ALPAMAYO2_SUPER_FIGURE_STYLE", "blog"),
        help="Visualization style for --save-viz/--save-json.",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=int(os.environ.get("ALPAMAYO2_SUPER_DIFFUSION_STEPS", "10")),
        help="Number of expert diffusion integration steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("ALPAMAYO2_SUPER_SEED", "42")),
        help="CUDA random seed.",
    )
    parser.add_argument(
        "--require-camera-projection",
        action="store_true",
        help="Fail if calibration is unavailable or no trajectory projects into camera views.",
    )
    args = parser.parse_args()

    if args.manifest and Path(args.manifest).is_file():
        sample = load_validation_sample(Path(args.manifest), args.sample_index)
        clip_id = sample["clip_id"]
        t0_us = sample["t0_us"]
    else:
        clip_id = args.clip_id
        try:
            t0_us = int(args.t0_us)
        except ValueError:
            parser.error("--t0-us must be an integer")

    try:
        run_smoke(
            model_id=args.model_id,
            clip_id=clip_id,
            t0_us=t0_us,
            save_viz=args.save_viz,
            save_json=args.save_json,
            figure_style=args.figure_style,
            diffusion_steps=args.diffusion_steps,
            seed=args.seed,
            require_camera_projection=args.require_camera_projection,
        )
    except (FileNotFoundError, NotADirectoryError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
