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

"""Tests for public imports used by the inference examples."""

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def run_python(code: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run a Python snippet with the project source tree on PYTHONPATH."""
    merged_env = os.environ.copy()
    merged_env["PYTHONPATH"] = str(SRC_ROOT)
    if env:
        merged_env.update(env)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=merged_env,
        text=True,
        capture_output=True,
        check=True,
    )


def test_notebook_import_surface() -> None:
    """Import the symbols used by the inference notebook without loading a checkpoint."""
    from alpamayo2_super import helper
    from alpamayo2_super.input_profiles import select_task_input
    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super
    from alpamayo2_super.text_tasks import (
        DEFAULT_GROUNDING_QUESTION,
        generate_text,
        prepare_text_generation_inputs,
        prepare_vqa_inputs,
        summarize_auto_labeling_conditioning,
    )
    from alpamayo2_super.visualization import (
        plot_blog_figure,
        plot_auto_labeling_result,
        plot_compact_inference_result,
        plot_grounding_result,
        plot_inference_result,
        plot_meta_action_result,
        plot_vqa_result,
    )

    assert Alpamayo2Super.__name__ == "Alpamayo2Super"
    assert helper.__name__ == "alpamayo2_super.helper"
    assert callable(select_task_input)
    assert callable(load_physical_aiavdataset)
    assert DEFAULT_GROUNDING_QUESTION.endswith("in JSON.")
    assert callable(generate_text)
    assert callable(prepare_text_generation_inputs)
    assert callable(prepare_vqa_inputs)
    assert callable(summarize_auto_labeling_conditioning)
    assert callable(plot_blog_figure)
    assert callable(plot_auto_labeling_result)
    assert callable(plot_compact_inference_result)
    assert callable(plot_grounding_result)
    assert callable(plot_inference_result)
    assert callable(plot_meta_action_result)
    assert callable(plot_vqa_result)


def test_dataset_loader_imports_from_release_package_only() -> None:
    """Keep the public data loader independent of the internal alpamayo package."""
    result = run_python(
        "from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset; "
        "print(load_physical_aiavdataset.__module__)"
    )

    assert result.stdout.strip() == "alpamayo2_super.load_physical_aiavdataset"


def test_top_level_import_does_not_import_torch() -> None:
    """Keep top-level package imports lightweight for CLI and metadata tooling."""
    result = run_python(
        "\n".join(
            [
                "import sys",
                "import alpamayo2_super",
                "print('TORCH_LOADED', 'torch' in sys.modules)",
            ]
        )
    )

    assert "TORCH_LOADED False" in result.stdout


def test_inference_help_is_lightweight_with_invalid_t0_env() -> None:
    """Help output should not import Torch or parse runtime-only env values."""
    env = {"ALPAMAYO2_SUPER_T0_US": "not-an-int"}
    for module_name in [
        "alpamayo2_super.inference_smoke",
        "alpamayo2_super.test_inference",
    ]:
        result = run_python(
            "\n".join(
                [
                    "import runpy",
                    "import sys",
                    f"sys.argv = ['{module_name}', '--help']",
                    "try:",
                    f"    runpy.run_module('{module_name}', run_name='__main__')",
                    "except SystemExit as exc:",
                    "    if exc.code not in (0, None):",
                    "        raise",
                    "print('TORCH_LOADED', 'torch' in sys.modules)",
                ]
            ),
            env=env,
        )

        assert "usage:" in result.stdout
        assert "TORCH_LOADED False" in result.stdout


def test_inference_help_lists_compact_figure_style_without_torch_import() -> None:
    """The public CLI advertises the compact debug figure style cheaply."""
    result = run_python(
        "\n".join(
            [
                "import runpy",
                "import sys",
                "sys.argv = ['alpamayo2_super.inference_smoke', '--help']",
                "try:",
                "    runpy.run_module('alpamayo2_super.inference_smoke', run_name='__main__')",
                "except SystemExit as exc:",
                "    if exc.code not in (0, None):",
                "        raise",
                "print('TORCH_LOADED', 'torch' in sys.modules)",
            ]
        )
    )

    assert "--figure-style" in result.stdout
    assert "{blog,compact}" in result.stdout
    assert "blog-7cam" not in result.stdout
    assert "compact" in result.stdout
    assert "TORCH_LOADED False" in result.stdout


def test_inference_smoke_selects_public_trajectory_profile() -> None:
    """The smoke CLI feeds the same validated camera profile as the notebook."""
    source = (SRC_ROOT / "alpamayo2_super" / "inference_smoke.py").read_text(encoding="utf-8")

    assert 'select_task_input(source_data, "trajectory")' in source


def test_text_task_notebooks_render_default_figures() -> None:
    """Each text-task notebook should call its default blog-style visualization helper."""
    notebook_expectations = {
        "meta_actions.ipynb": "plot_meta_action_result",
        "vqa.ipynb": "plot_vqa_result",
        "autolabeling.ipynb": "plot_auto_labeling_result",
    }
    for notebook_name, helper_name in notebook_expectations.items():
        notebook_path = PROJECT_ROOT / "notebooks" / notebook_name
        notebook_text = notebook_path.read_text(encoding="utf-8")

        assert helper_name in notebook_text
        assert "figure_path" in notebook_text
        assert "figure_metadata" in notebook_text
    vqa_notebook_text = (PROJECT_ROOT / "notebooks" / "vqa.ipynb").read_text(encoding="utf-8")
    assert "plot_grounding_result" in vqa_notebook_text
    assert "grounding_figure_path" in vqa_notebook_text
    auto_labeling_notebook = json.loads(
        (PROJECT_ROOT / "notebooks" / "autolabeling.ipynb").read_text(encoding="utf-8")
    )
    auto_labeling_source = "".join(
        line for cell in auto_labeling_notebook["cells"] for line in cell.get("source", [])
    )
    assert "video_path" in auto_labeling_source
    assert "Video" in auto_labeling_source
    assert "summarize_auto_labeling_conditioning" in auto_labeling_source
    assert '"conditioning"' in auto_labeling_source


def test_public_notebooks_apply_validated_task_input_profiles() -> None:
    """Every notebook selects its task-native six-camera/four-frame model input."""
    notebook_tasks = {
        "inference.ipynb": ("trajectory",),
        "meta_actions.ipynb": ("meta_action",),
        "autolabeling.ipynb": ("auto_labeling",),
        "vqa.ipynb": ("vqa", "grounding"),
    }
    for notebook_name, tasks in notebook_tasks.items():
        notebook = json.loads(
            (PROJECT_ROOT / "notebooks" / notebook_name).read_text(encoding="utf-8")
        )
        source = "".join(line for cell in notebook["cells"] for line in cell.get("source", []))

        assert "from alpamayo2_super.input_profiles import select_task_input" in source
        for task in tasks:
            assert f'select_task_input(source_data, "{task}")' in source


def test_public_model_id_is_consistent_across_entrypoints() -> None:
    """All user-facing entrypoints should default to the approved public model ID."""
    from alpamayo2_super.common.constants import PUBLIC_MODEL_ID
    from alpamayo2_super.inference_smoke import MODEL_ID

    assert PUBLIC_MODEL_ID == "nvidia/Alpamayo2-Super"
    assert MODEL_ID == PUBLIC_MODEL_ID
    for notebook_path in (PROJECT_ROOT / "notebooks").glob("*.ipynb"):
        notebook_text = notebook_path.read_text(encoding="utf-8")
        assert "PUBLIC_MODEL_ID" in notebook_text
        assert "nvidia/Alpamayo-2-Super-32B" not in notebook_text
    demo_text = (PROJECT_ROOT / "examples" / "two_gpu_nav_cfg_demo.py").read_text(encoding="utf-8")
    assert "from alpamayo2_super.common.constants import PUBLIC_MODEL_ID" in demo_text
    assert "DEFAULT_MODEL_ID = PUBLIC_MODEL_ID" in demo_text


def test_notebooks_resolve_relative_paths_from_project_root() -> None:
    """Notebook manifest and output overrides should not depend on kernel cwd."""
    from alpamayo2_super.inference_smoke import resolve_project_path

    project_root = Path("/repo")
    assert resolve_project_path("examples/validation_samples.json", project_root) == (
        project_root / "examples" / "validation_samples.json"
    )
    assert resolve_project_path("/tmp/outputs", project_root) == Path("/tmp/outputs")

    for notebook_path in (PROJECT_ROOT / "notebooks").glob("*.ipynb"):
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "".join(line for cell in notebook["cells"] for line in cell.get("source", []))
        assert "resolve_project_path" in source
        assert "MANIFEST = resolve_project_path(" in source
        assert "OUTPUT_DIR = resolve_project_path(" in source


def test_public_notebooks_only_require_clip_relative_timestamps() -> None:
    """Notebook users should not need an unavailable absolute clip origin."""
    for notebook_path in (PROJECT_ROOT / "notebooks").glob("*.ipynb"):
        notebook_text = notebook_path.read_text(encoding="utf-8")
        assert "clip_start_timestamp_us" not in notebook_text
        assert "ALPAMAYO2_SUPER_CLIP_START_TIMESTAMP_US" not in notebook_text


def test_dev_environment_includes_jupyter_frontend() -> None:
    """The documented notebook workflow should install a browser frontend."""
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert any(
        dependency.startswith("jupyterlab") for dependency in pyproject["dependency-groups"]["dev"]
    )


def test_readme_uses_configured_output_root_and_documents_vqa_grounding() -> None:
    """README commands should honor scratch output paths and describe both VQA passes."""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert '--save-viz "$ALPAMAYO2_SUPER_OUTPUT_DIR/' in readme
    assert '--save-json "$ALPAMAYO2_SUPER_OUTPUT_DIR/' in readme
    assert "--save-viz outputs/" not in readme
    assert "--save-json outputs/" not in readme
    assert "separate VQA and grounding" in readme
    assert "two PNG/JSON artifact pairs" in readme


def test_inference_preflight_validates_local_checkpoint_before_torch_import() -> None:
    """Missing local checkpoint paths should fail before importing Torch or touching CUDA."""
    result = run_python(
        "\n".join(
            [
                "import sys",
                "from alpamayo2_super.inference_smoke import run_smoke",
                "try:",
                "    run_smoke(model_id='/tmp/definitely-missing-alpamayo2-super')",
                "except FileNotFoundError as exc:",
                "    print('ERROR', exc)",
                "else:",
                "    raise AssertionError('expected missing checkpoint error')",
                "print('TORCH_LOADED', 'torch' in sys.modules)",
            ]
        )
    )

    assert "Local checkpoint path does not exist" in result.stdout
    assert "TORCH_LOADED False" in result.stdout


def test_inference_preflight_validates_local_checkpoint_files_before_torch_import(
    tmp_path: Path,
) -> None:
    """Incomplete local checkpoints should report the missing release files early."""
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text("{}", encoding="utf-8")

    result = run_python(
        "\n".join(
            [
                "import sys",
                "from alpamayo2_super.inference_smoke import run_smoke",
                "try:",
                f"    run_smoke(model_id={str(checkpoint_dir)!r})",
                "except FileNotFoundError as exc:",
                "    print('ERROR', exc)",
                "else:",
                "    raise AssertionError('expected incomplete checkpoint error')",
                "print('TORCH_LOADED', 'torch' in sys.modules)",
            ]
        )
    )

    assert "Local checkpoint is missing required file(s)" in result.stdout
    assert "tokenizer.json" in result.stdout
    assert "TORCH_LOADED False" in result.stdout


def test_inference_cli_reports_local_checkpoint_errors_without_traceback() -> None:
    """Expected local checkpoint errors should be argparse errors, not raw tracebacks."""
    result = run_python(
        "\n".join(
            [
                "import runpy",
                "import sys",
                "sys.argv = [",
                "    'alpamayo2_super.inference_smoke',",
                "    '--model-id',",
                "    '/tmp/definitely-missing-alpamayo2-super',",
                "]",
                "try:",
                "    runpy.run_module('alpamayo2_super.inference_smoke', run_name='__main__')",
                "except SystemExit as exc:",
                "    print('EXIT', exc.code)",
                "print('TORCH_LOADED', 'torch' in sys.modules)",
            ]
        )
    )

    assert "EXIT 2" in result.stdout
    assert "TORCH_LOADED False" in result.stdout
    assert "Local checkpoint path does not exist" in result.stderr
    assert "Traceback" not in result.stderr


def test_tokenizer_and_processor_enable_mistral_regex_fix(monkeypatch) -> None:
    """Tokenizer and processor loading should opt into the Transformers regex fix."""
    import alpamayo2_super.config as config_module
    from alpamayo2_super import helper

    tokenizer_kwargs: dict[str, object] = {}
    processor_kwargs: dict[str, object] = {}

    class FakeTokenizer:
        def add_tokens(self, tokens, special_tokens: bool = False):  # noqa: ANN001
            del tokens, special_tokens

    class FakeProcessor:
        tokenizer = None

    def fake_tokenizer_from_pretrained(name_or_path: str, **kwargs):
        del name_or_path
        tokenizer_kwargs.update(kwargs)
        return FakeTokenizer()

    def fake_processor_from_pretrained(name_or_path: str, **kwargs):
        del name_or_path
        processor_kwargs.update(kwargs)
        return FakeProcessor()

    monkeypatch.setattr(
        config_module.AutoTokenizer,
        "from_pretrained",
        fake_tokenizer_from_pretrained,
    )
    monkeypatch.setattr(
        helper.AutoProcessor,
        "from_pretrained",
        fake_processor_from_pretrained,
    )

    tokenizer = config_module.build_alpamayo2_super_tokenizer(
        "checkpoint",
        history_vocab_size=1,
        future_vocab_size=1,
    )
    processor = helper.get_processor(
        tokenizer,
        SimpleNamespace(_name_or_path="checkpoint", min_pixels=None, max_pixels=None),
    )

    assert tokenizer_kwargs["fix_mistral_regex"] is True
    assert processor_kwargs["fix_mistral_regex"] is True
    assert processor.tokenizer is tokenizer
