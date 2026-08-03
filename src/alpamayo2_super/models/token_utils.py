# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import re

import torch
from transformers import AutoTokenizer, StoppingCriteria

from alpamayo2_super.models.utils import SPECIAL_TOKENS

logger = logging.getLogger(__name__)


def extract_traj_tokens(
    output_tokens: torch.Tensor,
    special_token_ids: dict[str, int],
    tokens_per_future_traj: int,
    future_token_start_idx: int,
    traj_tokenizer_vocab_size: int,
) -> torch.Tensor:
    """Extract the trajectory tokens from the output tokens (parallel/vectorized version).

    This is a fully vectorized implementation that processes all batches in parallel
    without looping over the batch dimension.

    The output tokens to be [...<|cot_end|><|meta_action_start|>...<meta_action_end>
        <|future_traj_start|>]<|future_traj|>...<|future_traj_end|>.

    Args:
        output_tokens (torch.Tensor): The output tokens of shape [B, L].
        special_token_ids (dict[str, int]): Dictionary mapping special token names to their IDs.
        tokens_per_future_traj (int): Expected number of trajectory tokens.
        future_token_start_idx (int): The starting index for trajectory tokens in vocabulary.
        traj_tokenizer_vocab_size (int): Size of the trajectory tokenizer vocabulary.

    Returns:
        torch.Tensor: The trajectory tokens of shape [B, tokens_per_future_traj].
    """
    batch_size, seq_len = output_tokens.shape
    device = output_tokens.device

    # Initialize output tensor
    traj_tokens = torch.zeros(
        (batch_size, tokens_per_future_traj), dtype=output_tokens.dtype, device=device
    )

    # For each batch, find the first occurrence of end token
    # If no end token, use seq_len as the end position
    end_mask = output_tokens == special_token_ids["traj_future_end"]
    end_positions = torch.where(
        end_mask.any(dim=1),
        end_mask.int().argmax(dim=1),
        torch.full((batch_size,), seq_len, dtype=torch.long, device=device),
    )

    # For each batch, find the last occurrence of start token
    # We reverse the sequence to find the last occurrence
    start_mask = output_tokens == special_token_ids["traj_future_start"]
    start_mask_reversed = torch.flip(start_mask, dims=[1])
    last_start_positions_reversed = start_mask_reversed.int().argmax(dim=1)
    start_positions = seq_len - 1 - last_start_positions_reversed
    start_positions = torch.where(
        start_mask.any(dim=1),
        start_positions,
        torch.full((batch_size,), -1, dtype=torch.long, device=device),
    )

    # Create a range tensor for indexing [B, seq_len]
    range_tensor = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    valid_mask = (range_tensor > start_positions.unsqueeze(1)) & (
        range_tensor < end_positions.unsqueeze(1)
    )
    extracted_tokens = torch.where(valid_mask, output_tokens, torch.zeros_like(output_tokens))

    # Check for mismatches in token count
    n_valid_tokens = valid_mask.sum(dim=1)
    mismatch_mask = n_valid_tokens != tokens_per_future_traj
    if mismatch_mask.any():
        for idx in mismatch_mask.nonzero(as_tuple=True)[0]:
            logger.warning(
                f"Batch {idx}: Number of tokens is not equal to the expected number. "
                f"Expected: {tokens_per_future_traj}, Got: {n_valid_tokens[idx].item()}."
            )

    # Only gather from positions where we have valid tokens and within our output size
    cumsum_indices = torch.cumsum(valid_mask.int(), dim=1) - 1
    output_mask = valid_mask & (cumsum_indices < tokens_per_future_traj)

    if output_mask.any():
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
        output_positions = cumsum_indices[output_mask]
        batch_ids = batch_indices[output_mask]
        token_values = extracted_tokens[output_mask]
        token_values = token_values - future_token_start_idx

        # Check for invalid tokens
        invalid_tokens = (token_values < 0) | (token_values > traj_tokenizer_vocab_size)
        if invalid_tokens.any():
            logger.warning(f"Invalid token ids found in {invalid_tokens.sum().item()} positions.")

        # Clamp to valid range
        token_values = torch.clamp(token_values, min=0, max=traj_tokenizer_vocab_size - 1)
        traj_tokens[batch_ids, output_positions] = token_values

    return traj_tokens


_META_ACTION_SPLIT_RE = re.compile(r"(?P<head>(?:Longitudinal|Lateral|Lane)\s*:)")


def split_cot_and_meta_action(text: str) -> tuple[str, str]:
    """Split a no-special ``cot -> meta_action`` generation at the first axis marker."""
    if not text:
        return "", ""
    match = _META_ACTION_SPLIT_RE.search(text)
    if match is None:
        return text.rstrip(), ""
    return text[: match.start()].rstrip(), text[match.start() :].lstrip()


def extract_text_tokens(
    tokenizer: AutoTokenizer, output_tokens: torch.Tensor
) -> dict[str, list[str]]:
    """Extract no-special text outputs from assistant generations.

    Args:
        output_tokens (torch.Tensor): The output tokens of shape [B*ns*nj, L].

    Returns:
        dict[str, list[str]]: A dict containing raw outputs plus decoded text fields.
    """
    decoded_batch = tokenizer.batch_decode(output_tokens, skip_special_tokens=False)
    assistant_marker = "<|im_start|>assistant\n"
    traj_future_start = SPECIAL_TOKENS["traj_future_start"]
    im_end = "<|im_end|>"

    def assistant_text(text: str) -> str:
        if assistant_marker in text:
            text = text[text.rfind(assistant_marker) + len(assistant_marker) :]
        return text.split(traj_future_start, 1)[0].split(im_end, 1)[0].strip()

    assistant_texts = [assistant_text(text) for text in decoded_batch]
    extracted_text = {
        "raw_outputs": decoded_batch,
        "cot": [],
        "meta_action": [],
        "answer": [],
        "box": [],
        "cot_auto_labeling": [],
    }
    for text in assistant_texts:
        cot, meta_action = split_cot_and_meta_action(text)
        if meta_action:
            extracted_text["cot"].append(cot)
            extracted_text["meta_action"].append(meta_action)
            extracted_text["answer"].append(cot)
            extracted_text["box"].append("")
            extracted_text["cot_auto_labeling"].append("")
            continue

        extracted_text["cot"].append(text)
        extracted_text["meta_action"].append("")
        extracted_text["answer"].append(text)
        extracted_text["box"].append(text if _looks_like_grounding_box_text(text) else "")
        extracted_text["cot_auto_labeling"].append(text if text.lstrip().startswith("{") else "")
    return extracted_text


def _looks_like_grounding_box_text(text: str) -> bool:
    """Return True for no-special grounding box answer formats."""
    stripped = text.lstrip()
    if not stripped:
        return False
    if '"bbox_2d"' in stripped or '"bbox"' in stripped or '"box"' in stripped:
        return True
    return (
        stripped.startswith("[") and stripped.endswith("]") and any(ch.isdigit() for ch in stripped)
    )


class StopAfterEOS(StoppingCriteria):
    """Stopping criteria that stops the generation after one more token is generated
    after the first EOS token is generated.
    """

    def __init__(self, eos_token_id: int):
        """Args:
        eos_token_id (int): The EOS token ID.
        """
        self.eos_token_id = eos_token_id
        self.eos_found = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """Call the stopping criteria.

        Args:
            input_ids (torch.LongTensor): The input IDs of shape [B, L].
            scores (torch.FloatTensor): The scores of shape [B, L, vocab_size].

        Returns:
            bool: Whether to stop the generation.
        """
        batch_size = input_ids.shape[0]

        # Initialize tracking on first call
        if self.eos_found is None:
            self.eos_found = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        if self.eos_found.all():
            return True

        # Check which sequences have EOS in the last generated token
        last_tokens = input_ids[:, -1]
        current_has_eos = last_tokens == self.eos_token_id

        # Update which sequences just found EOS
        self.eos_found = self.eos_found | current_has_eos
        return False


def replace_padding_after_eos(
    token_ids: torch.Tensor,
    eos_token_id: int,
    pad_token_id: int,
) -> torch.Tensor:
    """Overwrite token IDs after the first EOS token with padding token ID.

    Args:
        token_ids (torch.Tensor): Token IDs of shape [B, L].
        eos_token_id (int): The end-of-sequence token ID to search for.
        pad_token_id (int): The padding token ID to use for masking.

    Returns:
        torch.Tensor: Token IDs with padding after the first EOS token of shape [B, L].

    Examples:
        >>> token_ids = torch.tensor([[1, 2, 3, 0, 5, 6], [7, 8, 0, 9, 0, 10]])
        >>> mask_after_eos(token_ids, eos_token_id=0, pad_token_id=-100)
        tensor([[   1,    2,    3,    0, -100, -100],
                [   7,    8,    0, -100, -100, -100]])
    """
    batch_size, seq_len = token_ids.shape

    # Find positions of EOS tokens
    eos_mask = token_ids == eos_token_id  # [B, L]

    # Get the position of the first EOS token in each sequence
    # Add seq_len where there's no EOS to handle sequences without EOS
    eos_positions = torch.where(
        eos_mask,
        torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1),
        torch.tensor(seq_len, device=token_ids.device),
    )
    first_eos_pos = eos_positions.min(dim=1, keepdim=True)[0]  # [B, 1]

    # Create a mask for positions after the first EOS
    position_indices = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)  # [1, L]
    mask_after = position_indices > first_eos_pos  # [B, L]

    # Apply padding inplace
    token_ids[mask_after] = pad_token_id
    return token_ids
