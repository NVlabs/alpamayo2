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

"""Base diffusion interface for expert trajectory inference."""

from abc import ABC, abstractmethod
from typing import Protocol

import torch
from torch import nn


class StepFn(Protocol):
    """Callable used by diffusion samplers to evaluate the expert denoiser."""

    def __call__(self, *, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict a denoising update for ``x`` at timestep ``t``."""
        ...


class BaseDiffusion(ABC, nn.Module):
    """Base class for diffusion samplers."""

    def __init__(self, x_dims: list[int] | tuple[int, ...] | int, *args, **kwargs) -> None:
        del args, kwargs
        super().__init__()
        self.x_dims = [x_dims] if isinstance(x_dims, int) else list(x_dims)

    @abstractmethod
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        step_fn: StepFn,
        device: torch.device | str = torch.device("cpu"),
        return_all_steps: bool = False,
        *args,
        unguided_step_fn: StepFn | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of actions from the diffusion model."""
        raise NotImplementedError
