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

"""Tests for the public two-GPU navigation CFG demo."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NAV_TEXT = "Turn left after the construction zone."
GENERATION_PROMPT = (
    "output the chain-of-thought reasoning of the driving process, then "
    "output the future trajectory."
)


class SpecialTokenAwareTokenizer:
    """Tokenize text while assigning explicit IDs to route delimiters."""

    ROUTE_TOKEN_IDS = {
        "<|route_start|>": 50_001,
        "<|route_end|>": 50_002,
    }

    def convert_tokens_to_ids(self, token: str) -> int:
        """Return the explicit ID for a route delimiter."""
        return self.ROUTE_TOKEN_IDS[token]

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        """Encode UTF-8 text while recognizing route delimiters as special tokens."""
        assert not add_special_tokens
        token_ids = []
        index = 0
        while index < len(text):
            for token, token_id in self.ROUTE_TOKEN_IDS.items():
                if text.startswith(token, index):
                    token_ids.append(token_id)
                    index += len(token)
                    break
            else:
                token_ids.extend(text[index].encode("utf-8"))
                index += 1
        return token_ids


class RecordingProcessor:
    """Render real conversation messages and tokenize with explicit route IDs."""

    def __init__(self) -> None:
        self.rendered_texts: list[str] = []
        self.tokenizer = SpecialTokenAwareTokenizer()

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        add_vision_id: bool,
    ) -> str:
        """Flatten the multimodal messages into a stable prompt string."""
        assert not tokenize
        assert add_generation_prompt
        assert not add_vision_id
        rendered = "".join(
            entry["text"] if entry["type"] == "text" else "<image>"
            for message in messages
            for entry in message["content"]
        )
        self.rendered_texts.append(rendered)
        return rendered

    def __call__(
        self,
        *,
        text: str,
        images: torch.Tensor,
        videos: None,
        padding: bool,
        return_tensors: str,
        do_rescale: bool,
    ) -> dict[str, torch.Tensor]:
        """Encode prompt bytes so tests can inspect the tokenized prompt."""
        assert videos is None
        assert not padding
        assert return_tensors == "pt"
        assert not do_rescale
        input_ids = torch.tensor(
            [self.tokenizer.encode(text, add_special_tokens=False)],
            dtype=torch.long,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": images,
            "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }


def load_example_module() -> ModuleType:
    """Load the example module without requiring the package to be installed."""
    module_name = "two_gpu_nav_cfg_demo"
    module_path = PROJECT_ROOT / "examples" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _sample_data() -> dict:
    """Build the minimal PhysicalAI-style sample used by prompt preparation."""
    return {
        "image_frames": torch.zeros((1, 1, 3, 2, 2), dtype=torch.uint8),
        "camera_indices": torch.tensor([0], dtype=torch.int64),
        "ego_history_xyz": torch.zeros((1, 1, 2, 3), dtype=torch.float32),
        "ego_history_rot": torch.eye(3).view(1, 1, 1, 3, 3).repeat(1, 1, 2, 1, 1),
    }


def _model() -> SimpleNamespace:
    """Build the model surface consumed by prompt preparation."""
    return SimpleNamespace(
        tokenizer=object(),
        config=SimpleNamespace(
            tokens_per_history_traj=2,
            tokens_per_future_traj=3,
            include_camera_ids=False,
            frame_label="frame_num",
        ),
    )


def _subsequence_start(token_ids: list[int], subsequence: list[int]) -> int:
    """Return the first index of ``subsequence`` or -1 when it is absent."""
    width = len(subsequence)
    return next(
        (
            index
            for index in range(len(token_ids) - width + 1)
            if token_ids[index : index + width] == subsequence
        ),
        -1,
    )


def _prepare_prompts(monkeypatch) -> tuple[dict, RecordingProcessor]:
    """Prepare guided and unguided prompts with the recording processor."""
    module = load_example_module()
    processor = RecordingProcessor()
    monkeypatch.setattr(
        "alpamayo2_super.helper.get_processor",
        lambda tokenizer, model_config: processor,
    )
    return module.prepare_nav_model_inputs(_sample_data(), _model(), NAV_TEXT), processor


def test_prepare_nav_model_inputs_renders_raw_guided_navigation(monkeypatch) -> None:
    """The guided prompt matches no-special train/eval ordering and raw navigation text."""
    _, processor = _prepare_prompts(monkeypatch)

    rendered = processor.rendered_texts[0]

    assert NAV_TEXT in rendered
    assert "<|route_start|>" not in rendered
    assert "<|route_end|>" not in rendered
    assert rendered.index("<image>") < rendered.index("<|traj_history_start|>")
    assert rendered.index("<|traj_history_start|>") < rendered.index(NAV_TEXT)
    assert rendered.index(NAV_TEXT) < rendered.index(GENERATION_PROMPT)


def test_prepare_nav_model_inputs_tokenized_guided_prompt_has_no_route_ids(monkeypatch) -> None:
    """Tokenized navigation stays ordered without route delimiter special-token IDs."""
    model_inputs, processor = _prepare_prompts(monkeypatch)
    tokenizer = processor.tokenizer
    token_ids = model_inputs["tokenized_data"]["input_ids"][0].tolist()
    route_start_id = tokenizer.convert_tokens_to_ids("<|route_start|>")
    route_end_id = tokenizer.convert_tokens_to_ids("<|route_end|>")

    assert tokenizer.encode("<|route_start|>", add_special_tokens=False) == [route_start_id]
    assert tokenizer.encode("<|route_end|>", add_special_tokens=False) == [route_end_id]
    assert route_start_id not in token_ids
    assert route_end_id not in token_ids

    history_end_tokens = tokenizer.encode(
        "<|traj_history_end|>",
        add_special_tokens=False,
    )
    history_end = _subsequence_start(
        token_ids,
        history_end_tokens,
    )
    navigation = _subsequence_start(
        token_ids,
        tokenizer.encode(NAV_TEXT, add_special_tokens=False),
    )
    prompt_start = _subsequence_start(
        token_ids,
        tokenizer.encode(GENERATION_PROMPT, add_special_tokens=False),
    )
    assert 0 <= history_end
    assert history_end + len(history_end_tokens) <= navigation < prompt_start


def test_prepare_nav_model_inputs_independently_renders_unguided_prompt(monkeypatch) -> None:
    """The unguided prompt is independently rendered without navigation conditioning."""
    _, processor = _prepare_prompts(monkeypatch)

    assert len(processor.rendered_texts) == 2
    unguided_rendered = processor.rendered_texts[1]

    assert NAV_TEXT not in unguided_rendered
    assert GENERATION_PROMPT in unguided_rendered
    assert unguided_rendered.index("<image>") < unguided_rendered.index("<|traj_history_start|>")
    assert unguided_rendered.index("<|traj_history_start|>") < unguided_rendered.index(
        GENERATION_PROMPT
    )


def test_prepare_nav_model_inputs_tokenizes_unguided_prompt_without_navigation(monkeypatch) -> None:
    """The independently tokenized unguided prompt omits navigation token IDs."""
    model_inputs, processor = _prepare_prompts(monkeypatch)
    token_ids = model_inputs["unguided_tokenized_data"]["input_ids"][0].tolist()
    tokenizer = processor.tokenizer

    assert (
        _subsequence_start(
            token_ids,
            tokenizer.encode(NAV_TEXT, add_special_tokens=False),
        )
        == -1
    )
    assert (
        _subsequence_start(
            token_ids,
            tokenizer.encode(GENERATION_PROMPT, add_special_tokens=False),
        )
        >= 0
    )


def test_unguided_prefill_reuses_guided_visual_payload(monkeypatch) -> None:
    """The independently tokenized unguided branch should not own duplicate visual tensors."""
    module = load_example_module()
    model_inputs, _ = _prepare_prompts(monkeypatch)
    guided = model_inputs["tokenized_data"]
    unguided = model_inputs["unguided_tokenized_data"]

    assert "pixel_values" not in unguided
    assert "image_grid_thw" not in unguided

    prefill_inputs = module.build_unguided_prefill_inputs(
        tokenized_data=guided,
        unguided_tokenized_data=unguided,
        pad_token_id=0,
    )

    assert prefill_inputs["input_ids"] is unguided["input_ids"]
    assert prefill_inputs["attention_mask"] is unguided["attention_mask"]
    assert prefill_inputs["pixel_values"] is guided["pixel_values"]
    assert prefill_inputs["image_grid_thw"] is guided["image_grid_thw"]


def test_unguided_continuation_retains_guided_cot_with_unguided_positions() -> None:
    """Unguided cache extension replays all guided tokens after its shorter prefix."""
    module = load_example_module()
    guided_sequences = torch.tensor([[10, 11, 701, 702, 0]], dtype=torch.long)
    unguided_input_ids = torch.tensor([[20, 21, 22]], dtype=torch.long)
    unguided_prefix_mask = torch.tensor([[0, 1, 1]], dtype=torch.long)

    continuation = module.build_unguided_continuation_inputs(
        guided_sequences=guided_sequences,
        guided_prefix_length=2,
        unguided_input_ids=unguided_input_ids,
        unguided_prefix_mask=unguided_prefix_mask,
        pad_token_id=0,
        num_traj_samples=1,
    )

    assert continuation["input_ids"].tolist() == [[701, 702, 0]]
    assert continuation["attention_mask"].tolist() == [[0, 1, 1, 1, 1, 0]]
    assert continuation["cache_position"].tolist() == [3, 4, 5]
    assert continuation["full_sequences"].tolist() == [[20, 21, 22, 701, 702, 0]]


def test_move_plain_tensor_attrs_moves_unregistered_nested_tensors() -> None:
    """Manual placement should include tensors that are not registered buffers."""
    module = load_example_module()

    class Child(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tensor_attr = torch.ones(2)
            self.tensor_list = [torch.zeros(1), "keep"]
            self.tensor_tuple = (torch.arange(2), "keep")

    parent = torch.nn.Module()
    parent.child = Child()

    module.move_plain_tensor_attrs_to_device(parent, torch.device("cpu"))

    assert parent.child.tensor_attr.device.type == "cpu"
    assert parent.child.tensor_list[0].device.type == "cpu"
    assert parent.child.tensor_list[1] == "keep"
    assert parent.child.tensor_tuple[0].device.type == "cpu"
    assert parent.child.tensor_tuple[1] == "keep"


def test_move_cache_to_device_moves_layer_tensor_attrs() -> None:
    """Guided and unguided KV cache tensors should move as a unit."""
    module = load_example_module()

    class Layer:
        def __init__(self) -> None:
            self.key = torch.ones(1)
            self.values = [torch.zeros(1)]
            self.extra = "keep"

    class Cache:
        def __init__(self) -> None:
            self.layers = [Layer()]

    cache = Cache()
    returned = module.move_cache_to_device(cache, torch.device("cpu"))

    assert returned is cache
    assert cache.layers[0].key.device.type == "cpu"
    assert cache.layers[0].values[0].device.type == "cpu"
    assert cache.layers[0].extra == "keep"


def test_demo_delegates_cfg_integration_to_release_sampler() -> None:
    """The two-GPU example should not maintain a second Euler implementation."""
    source = (PROJECT_ROOT / "examples" / "two_gpu_nav_cfg_demo.py").read_text(encoding="utf-8")
    sample_body = source[source.index("def sample_with_nav_cfg") : source.index("def _first_cot")]

    assert "model.expert.diffusion.sample(" in sample_body
    assert "unguided_step_fn=" in sample_body
    assert "torch.linspace(" not in sample_body
