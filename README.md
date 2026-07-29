<div align="center">

# Alpamayo 2 Super

### 34B Multi-Task Autonomous Vehicle Foundation Model

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](./LICENSE)

</div>

Alpamayo 2 Super is a 34-billion parameter foundation model designed to tackle multiple autonomous
vehicle (AV) development tasks. It combines a 32B VLM backbone with a 2B diffusion expert. The
released inference path uses the trained VLM backbone to generate Chain-of-Causation text, then
samples future trajectories through the trained action expert. The CLI, notebook, and importable
APIs in this repository run the real expert model.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Model And Data Access](#model-and-data-access)
- [CLI Inference](#cli-inference)
- [Advanced Two-GPU Navigation CFG Demo](#advanced-two-gpu-navigation-cfg-demo)
- [Notebook Inference](#notebook-inference)
- [Text Task Notebooks](#text-task-notebooks)
- [Visualization API](#visualization-api)
- [Tests](#tests)
- [Project Structure](#project-structure)
- [Support](#support)
- [License](#license)

## Prerequisites

- Linux host with an NVIDIA GPU, CUDA driver/runtime, and enough VRAM for a 34B model.
- CUDA Toolkit 12.x with `nvcc`; `flash-attn` builds during installation.
- Python 3.12.
- [`uv`](https://docs.astral.sh/uv/) for environment management.
- Access to the gated Alpamayo 2 Super model and
  [`nvidia/PhysicalAI-Autonomous-Vehicles`](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
  dataset on Hugging Face.

Set cache directories before install or inference if your home directory is small:

```bash
export HF_HOME=/path/to/hf-cache
export UV_CACHE_DIR=/path/to/uv-cache
export MPLCONFIGDIR=/tmp/alpamayo2_super_mpl
```

## Setup

Run setup from the repository root, the directory that contains this `README.md`:

```bash
export UV_PROJECT_ENVIRONMENT=.venv
uv sync --locked --dev
source "${UV_PROJECT_ENVIRONMENT}/bin/activate"
hf auth login --token "$HF_TOKEN"
```

`UV_PROJECT_ENVIRONMENT=.venv` keeps `uv sync` pointed at this project's managed environment
even if another virtual environment is already active. If you prefer interactive login, run
`hf auth login` after activating the environment.

For a read-only checkout or scratch review, put the environment and outputs under `/tmp`:

```bash
export UV_PROJECT_ENVIRONMENT=/tmp/alpamayo2_super_venv
export UV_CACHE_DIR=/tmp/alpamayo2_super_uv_cache
export HF_HOME=/tmp/alpamayo2_super_hf_cache
export ALPAMAYO2_SUPER_OUTPUT_DIR=/tmp/alpamayo2_super_outputs
uv sync --locked --dev
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
hf auth login --token "$HF_TOKEN"
```

Initial setup downloads large CUDA/PyTorch wheels and builds `flash-attn`. If that build fails,
check that the CUDA Toolkit, `nvcc`, and a compatible compiler are visible in the same shell; set
`MAX_JOBS` lower if the build exhausts host memory.

Quick checks:

```bash
hf auth whoami
python -m alpamayo2_super.inference_smoke --help
python - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

## Model And Data Access

For the public release, use the Hugging Face model id:

```bash
export ALPAMAYO2_SUPER_MODEL_ID=nvidia/Alpamayo2-Super
```

During local validation you can point to a release-native checkpoint directory instead:

```bash
export ALPAMAYO2_SUPER_MODEL_ID=/path/to/alpamayo2-super
```

A local checkpoint directory must contain `config.json`, tokenizer/processor files, and one or
more `*.safetensors` weight shards. The CLI validates this layout before loading CUDA.

The default examples use `examples/validation_samples.json`, which references PhysicalAI-AV clip
IDs. You must have dataset access accepted for the same Hugging Face account used by
`hf auth login`. The PhysicalAI-AV package will download the required metadata and clip assets from
`nvidia/PhysicalAI-Autonomous-Vehicles` on first use.

## CLI Inference

The smoke script downloads one PhysicalAI-AV clip, runs one sampled expert trajectory, prints the
generated CoT and trajectory metrics, and writes the default task-native six-camera visualization:

```bash
export ALPAMAYO2_SUPER_OUTPUT_DIR="${ALPAMAYO2_SUPER_OUTPUT_DIR:-outputs}"
mkdir -p "$ALPAMAYO2_SUPER_OUTPUT_DIR"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m alpamayo2_super.inference_smoke \
  --model-id "$ALPAMAYO2_SUPER_MODEL_ID" \
  --manifest examples/validation_samples.json \
  --sample-index 0 \
  --save-viz "$ALPAMAYO2_SUPER_OUTPUT_DIR/sample0.png" \
  --save-json "$ALPAMAYO2_SUPER_OUTPUT_DIR/sample0.json"
```

Successful output includes:

- `Chain-of-Causation`, the model's generated reasoning text.
- `minADE`, a trajectory-distance diagnostic against the sample ground truth.
- `figure_style`, the visualization style used for the saved PNG/JSON.
- `projection_available`, whether camera overlays were rendered from calibration.
- `$ALPAMAYO2_SUPER_OUTPUT_DIR/sample0.png` and
  `$ALPAMAYO2_SUPER_OUTPUT_DIR/sample0.json`.

To reproduce the six-camera figure shown in the Alpamayo 2 Super technical blog, use the same
smoke path. The checked-in default sample is PhysicalAI-AV clip
`030c760c-ae38-49aa-9ad8-f5650a545d26` at `t0_us=5100000`.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m alpamayo2_super.inference_smoke \
  --model-id "$ALPAMAYO2_SUPER_MODEL_ID" \
  --manifest examples/validation_samples.json \
  --sample-index 0 \
  --require-camera-projection \
  --save-viz "$ALPAMAYO2_SUPER_OUTPUT_DIR/blog.png" \
  --save-json "$ALPAMAYO2_SUPER_OUTPUT_DIR/blog.json"
```

For a smaller debug figure, pass `--figure-style compact`.

To run the small checked-in parity subset, switch the manifest:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m alpamayo2_super.inference_smoke \
  --model-id "$ALPAMAYO2_SUPER_MODEL_ID" \
  --manifest examples/public_golden_validation_samples.json \
  --sample-index 0 \
  --save-viz "$ALPAMAYO2_SUPER_OUTPUT_DIR/public_golden_sample0.png" \
  --save-json "$ALPAMAYO2_SUPER_OUTPUT_DIR/public_golden_sample0.json"
```

## Advanced Two-GPU Navigation CFG Demo

`examples/two_gpu_nav_cfg_demo.py` demonstrates navigation classifier-free guidance with the
manual placement validated for two 80GB H100 GPUs: VLM generation on `cuda:0`, then expert
denoising plus guided/unguided KV caches on `cuda:1`. This is an advanced demo, not the default
public API. The regular CLI and notebook above run the standard single-prompt expert inference
path.

Use explicit two-GPU placement rather than Hugging Face `device_map="auto"` or
`device_map="balanced"` for this CFG path:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python examples/two_gpu_nav_cfg_demo.py \
  --model-id "$ALPAMAYO2_SUPER_MODEL_ID" \
  --manifest examples/validation_samples.json \
  --sample-index 0 \
  --save-viz "$ALPAMAYO2_SUPER_OUTPUT_DIR/nav_cfg_sample0.png" \
  --save-json "$ALPAMAYO2_SUPER_OUTPUT_DIR/nav_cfg_sample0.json"
```

The demo samples one trajectory with 10 diffusion steps by default and records the navigation text,
CFG guidance weight, device placement, decoded CoT, trajectory metrics, and visualization metadata
in the JSON sidecar. On the validated checkpoint, the 7-camera x 4-frame run peaked at roughly
67 GiB on the VLM GPU and 71 GiB on the expert GPU. Host memory must also be sufficient because the
demo loads the model before manually placing VLM and expert modules.

## Notebook Inference

The notebook version is available at `notebooks/inference.ipynb`. It uses the same default
manifest, environment variables, model loading, generation settings, and visualization API as the
CLI. Before model preparation, the notebook selects the validated trajectory profile: four frames
from camera IDs `[0, 1, 2, 3, 5, 6]`.

```bash
source "${UV_PROJECT_ENVIRONMENT:-.venv}/bin/activate"
python -m ipykernel install --user --name alpamayo2-super --display-name "Alpamayo 2 Super"
export ALPAMAYO2_SUPER_MODEL_ID=/path/to/alpamayo2-super
export ALPAMAYO2_SUPER_OUTPUT_DIR=outputs/notebook
jupyter lab
```

Open `notebooks/inference.ipynb` from the repository root and select the `Alpamayo 2 Super` kernel.
The notebook writes the same PNG and JSON artifact pair as the CLI. Set these optional environment
variables before opening the notebook to choose a sample:

```bash
export ALPAMAYO2_SUPER_VALIDATION_MANIFEST=examples/validation_samples.json
export ALPAMAYO2_SUPER_SAMPLE_INDEX=0
export ALPAMAYO2_SUPER_DIFFUSION_STEPS=10
export ALPAMAYO2_SUPER_NUM_TRAJ_SAMPLES=1
```

## Text Task Notebooks

Three additional notebooks demonstrate VLM text tasks trained on Alpamayo 2 Super:

- `notebooks/meta_actions.ipynb` generates Chain-of-Causation text followed by a meta-action
  block with longitudinal, lateral, and lane actions.
- `notebooks/autolabeling.ipynb` generates the four-field auto-labeling JSON schema:
  `critical_components_analysis`, `ego_vehicle_motion_analysis`, `trajectory_analysis`, and
  `chain_of_causation`.
- `notebooks/vqa.ipynb` performs separate VQA and grounding generations using the no-special
  prompt format used by Alpax text eval. It writes two PNG/JSON artifact pairs: one for the
  free-form scene answer and one for generated grounding coordinates and overlays.

Each notebook loads the canonical seven-camera PhysicalAI-AV sample, then calls the public
`select_task_input(...)` helper before model preparation. The fixed six-camera/four-frame profiles
are:

- VQA: camera IDs `[0, 1, 2, 3, 4, 5]`.
- Trajectory, meta-action, auto-labeling, and grounding: camera IDs `[0, 1, 2, 3, 5, 6]`.

The input-profile record is retained in visualization metadata so the camera names, IDs, and frame
indices travel with each result.

The auto-labeling notebook follows the model's training contract: four synchronized camera
context frames ending at `t0`, ego trajectory history through `t0`, and a future ego trajectory.
It does not pass post-`t0` camera frames to the model. The offline teacher used to create the
training labels may inspect future video, but that future video is not an Alpamayo 2 Super
auto-labeling input. By default the notebook uses the validation sample's future ego trajectory.
To first sample an expert trajectory and auto-label that predicted future, set:

```bash
export ALPAMAYO2_SUPER_AUTOLABEL_FUTURE_SOURCE=predicted
```

All three notebooks load `ALPAMAYO2_SUPER_MODEL_ID` and use
`examples/validation_samples.json` by default. Meta-action writes task JSON plus a blog-style PNG
to `ALPAMAYO2_SUPER_OUTPUT_DIR`. VQA writes separate VQA and grounding PNG/JSON pairs.
Auto-labeling writes task JSON, a synchronized six-camera context MP4, and a PNG poster suitable
for a blog video thumbnail. The MP4 animates the same four frames consumed by the model, nominally
sampled at `t0 - 0.3 s`, `t0 - 0.2 s`, `t0 - 0.1 s`, and `t0`; it is not a future rollout. The JSON
payload records the decoded timestamp offsets in the conditioning summary, plus artifact paths and
visualization metadata, so downstream review tools can pair each model output with the exact
camera/frame layout.

## Visualization API

The visualization API is importable:

```python
from alpamayo2_super.visualization import (
    plot_auto_labeling_result,
    plot_compact_inference_result,
    plot_grounding_result,
    plot_inference_result,
    plot_meta_action_result,
    plot_vqa_result,
)
```

`plot_inference_result(...)` renders the public task-native six-camera layout used by the technical
blog. The upper row is cross left, front wide, and cross right. The lower row is rear left, front
tele, and rear right for driving tasks; VQA uses rear tele in the lower-center position instead.
The camera grid is followed by a full-width BEV trajectory comparison and predicted CoC panel. It
also renders camera trajectory overlays when PhysicalAI-AV calibration is available. Each overlay
is a calibrated 2.0-meter-wide ground-plane corridor that approximates vehicle width, not model
uncertainty. The NVIDIA-green prediction is drawn above dashed-magenta ground truth, with blue
reserved for ego history. If calibration is missing,
visualization falls back to camera frames + BEV + CoC and records `"projection_available": false`
in the JSON sidecar.

`plot_compact_inference_result(...)` renders a smaller four-camera debug layout for quick local
inspection.

`plot_meta_action_result(...)`, `plot_vqa_result(...)`, `plot_grounding_result(...)`, and
`plot_auto_labeling_result(...)` render the default text-task notebook figures from the selected
task profile and add full-width panels for generated meta-action fields, plain question/answer
text, generated grounding boxes, or four-field auto-labeling JSON. The grounding helper overlays
generated boxes on camera images when the answer contains `bbox_2d` / `bbox` / `box` JSON and
prints the predicted coordinates in the result panel. The auto-labeling helper animates the
complete six-camera/four-frame context input at two frames per second in a synchronized MP4. Its
PNG poster retains a filmstrip of all four frames for static review and blog previews.

Use the PNG and JSON together when judging an inference result: check that the camera frames match
the driving scene described by the CoT, the predicted trajectory is plausible in BEV, and projected
overlays agree with the road geometry when `projection_available` is true.

## Inference Smoke Check

```bash
python -m alpamayo2_super.test_inference --help
```

`test_inference.py` is a compatibility entry point for the same end-to-end inference smoke
documented above. Running inference requires a GPU, model access, and PhysicalAI-AV access.

## Project Structure

```text
alpamayo-2-super/
|-- examples/
|   |-- public_golden_validation_samples.json
|   |-- two_gpu_nav_cfg_demo.py
|   `-- validation_samples.json
|-- notebooks/
|   |-- clip_ids.parquet
|   |-- autolabeling.ipynb
|   |-- inference.ipynb
|   |-- meta_actions.ipynb
|   `-- vqa.ipynb
|-- src/
|   `-- alpamayo2_super/
|       |-- action_space/
|       |-- chat_template/
|       |-- common/
|       |-- diffusion/
|       |-- geometry/
|       |-- models/
|       |   |-- action_in_proj.py
|       |   |-- alpamayo2_super.py
|       |   |-- expert.py
|       |   `-- expert_utils.py
|       |-- config.py
|       |-- helper.py
|       |-- input_profiles.py
|       |-- inference_smoke.py
|       |-- load_physical_aiavdataset.py
|       |-- text_tasks.py
|       |-- test_inference.py
|       |-- visualization.py
|       `-- viz_utils.py
|-- .github/ISSUE_TEMPLATE/
|-- CONTRIBUTING.md
|-- LICENSE
|-- SECURITY.md
|-- pyproject.toml
`-- uv.lock
```

## Support

📣 **Usage questions and discussion about Alpamayo 2 Super**: please join us on the [Alpamayo NV Developer Forum](https://forums.developer.nvidia.com/c/autonomous-vehicles/alpamayo/766).

🐛 **Code-level bugs, documentation issues, and feature requests**: file a [GitHub issue](../../issues/new/choose) using the appropriate template (Bug report, Documentation request, or Feature request). The relevant NVIDIA responder is auto-assigned via the `assignees:` field on the template.

🚨 **Security vulnerabilities**: please use [NVIDIA's Vulnerability Disclosure Program](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). Do not file security issues publicly here.

## License

Apache License 2.0 - see [LICENSE](./LICENSE) for details.
