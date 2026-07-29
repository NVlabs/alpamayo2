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

"""Tests for Alpamayo 2 Super model utilities."""

import pytest
import torch

from alpamayo2_super.models.utils import replace_pad_token


PAD_ID = 99


def test_replace_pad_token_replaces_each_row_with_its_encoded_ids() -> None:
    """Each row's placeholders receive only that row's encoded ids."""
    input_ids = torch.tensor(
        [
            [PAD_ID, 1, PAD_ID, 2],
            [3, PAD_ID, 4, PAD_ID],
        ]
    )
    new_ids = torch.tensor(
        [
            [10, 11],
            [20, 21],
        ]
    )

    result = replace_pad_token(input_ids, new_ids, PAD_ID)

    torch.testing.assert_close(
        result,
        torch.tensor(
            [
                [10, 1, 11, 2],
                [3, 20, 4, 21],
            ]
        ),
    )


def test_replace_pad_token_rejects_too_few_encoded_ids() -> None:
    """A source row shorter than its placeholder span raises a clear ValueError."""
    input_ids = torch.tensor(
        [
            [PAD_ID, PAD_ID, PAD_ID],
            [PAD_ID, PAD_ID, PAD_ID],
        ]
    )
    new_ids = torch.tensor(
        [
            [10, 11],
            [20, 21],
        ]
    )

    with pytest.raises(ValueError) as exc_info:
        replace_pad_token(input_ids, new_ids, PAD_ID)

    message = str(exc_info.value)
    assert "placeholder counts=[3, 3]" in message
    assert "new_ids tokens per row=2" in message


def test_replace_pad_token_rejects_too_many_encoded_ids() -> None:
    """A source row longer than its placeholder span is not silently truncated."""
    input_ids = torch.tensor(
        [
            [PAD_ID, 1, PAD_ID],
            [PAD_ID, 2, PAD_ID],
        ]
    )
    new_ids = torch.tensor(
        [
            [10, 11, 12],
            [20, 21, 22],
        ]
    )

    with pytest.raises(ValueError) as exc_info:
        replace_pad_token(input_ids, new_ids, PAD_ID)

    message = str(exc_info.value)
    assert "placeholder counts=[2, 2]" in message
    assert "new_ids tokens per row=3" in message


def test_replace_pad_token_rejects_per_row_mismatch_when_totals_match() -> None:
    """Aggregate equality cannot hide row-local placeholder mismatches."""
    input_ids = torch.tensor(
        [
            [PAD_ID, 1, 2, 3],
            [PAD_ID, PAD_ID, PAD_ID, 4],
        ]
    )
    new_ids = torch.tensor(
        [
            [10, 11],
            [20, 21],
        ]
    )

    with pytest.raises(ValueError) as exc_info:
        replace_pad_token(input_ids, new_ids, PAD_ID)

    message = str(exc_info.value)
    assert "mismatched rows=[0, 1]" in message
    assert "placeholder counts=[1, 3]" in message
    assert "new_ids tokens per row=2" in message


@pytest.mark.parametrize(
    ("input_ids", "new_ids"),
    [
        pytest.param(
            torch.tensor([PAD_ID, PAD_ID]),
            torch.tensor([[10, 11]]),
            id="rank-one-input-ids",
        ),
        pytest.param(
            torch.tensor([[PAD_ID, PAD_ID]]),
            torch.tensor([10, 11]),
            id="rank-one-new-ids",
        ),
    ],
)
def test_replace_pad_token_rejects_non_matrix_inputs(
    input_ids: torch.Tensor,
    new_ids: torch.Tensor,
) -> None:
    """Row-local replacement requires rank-two input and replacement tensors."""
    with pytest.raises(ValueError, match="rank-2"):
        replace_pad_token(input_ids, new_ids, PAD_ID)


def test_replace_pad_token_rejects_different_batch_sizes() -> None:
    """Input and replacement tensors must describe the same batch."""
    input_ids = torch.tensor(
        [
            [PAD_ID, PAD_ID],
            [PAD_ID, PAD_ID],
        ]
    )
    new_ids = torch.tensor([[10, 11]])

    with pytest.raises(ValueError, match="same batch size"):
        replace_pad_token(input_ids, new_ids, PAD_ID)
