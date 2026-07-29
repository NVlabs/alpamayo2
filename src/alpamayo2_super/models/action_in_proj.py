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

"""Action-to-token projection modules for the expert denoiser."""

import math

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMS normalization used by the action projection MLP."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension of ``x``."""
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return output.type_as(x) * self.weight


class MLPEncoder(nn.Module):
    """MLP encoder used after Fourier feature expansion."""

    def __init__(
        self,
        num_input_feats: int,
        num_enc_layers: int,
        hidden_size: int,
        outdim: int,
    ) -> None:
        super().__init__()
        if num_enc_layers < 1:
            raise ValueError(f"num_enc_layers must be >= 1, got {num_enc_layers}")

        enc_layers: list[nn.Module] = [nn.Linear(num_input_feats, hidden_size), nn.SiLU()]
        for layer_idx in range(num_enc_layers):
            if layer_idx < num_enc_layers - 1:
                enc_layers.extend(
                    [
                        RMSNorm(hidden_size, eps=1e-5),
                        nn.Linear(hidden_size, hidden_size),
                        nn.SiLU(),
                    ]
                )
            else:
                enc_layers.extend(
                    [
                        RMSNorm(hidden_size, eps=1e-5),
                        nn.Linear(hidden_size, outdim),
                    ]
                )
        self.trunk = nn.Sequential(*enc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, C)`` features into ``(B, outdim)``."""
        return self.trunk(x)


class FourierEncoderV2(nn.Module):
    """Fourier feature encoder with logarithmically spaced frequencies."""

    def __init__(self, dim: int, max_freq: float = 100.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        half = dim // 2
        freqs = torch.logspace(0, math.log10(max_freq), steps=half)
        self.out_dim = dim
        self.register_buffer("freqs", freqs[None, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode scalar inputs into sinusoidal features."""
        arg = x[..., None] * self.freqs * 2 * torch.pi
        return torch.cat([torch.sin(arg), torch.cos(arg)], -1) * math.sqrt(2)


class PerWaypointActionInProjV2(nn.Module):
    """Project per-waypoint action and diffusion timestep features to expert tokens."""

    def __init__(
        self,
        in_dims: list[int],
        out_dim: int,
        num_enc_layers: int = 4,
        hidden_size: int = 1024,
        max_freq: float = 100.0,
        num_fourier_feats: int = 20,
    ) -> None:
        super().__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        self.sinus = nn.ModuleList(
            FourierEncoderV2(dim=num_fourier_feats, max_freq=max_freq) for _ in range(in_dims[-1])
        )
        self.timestep_fourier_encoder = FourierEncoderV2(
            dim=num_fourier_feats,
            max_freq=max_freq,
        )
        num_input_feats = (
            sum(encoder.out_dim for encoder in self.sinus) + self.timestep_fourier_encoder.out_dim
        )
        self.encoder = MLPEncoder(
            num_input_feats=num_input_feats,
            num_enc_layers=num_enc_layers,
            hidden_size=hidden_size,
            outdim=out_dim,
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Project actions of shape ``(B, T, action_dim)`` to expert token embeddings."""
        batch_size, n_waypoints, _ = x.shape

        x = x.float()
        timesteps = timesteps.float()

        action_feats = torch.cat(
            [encoder(x[:, :, idx]) for idx, encoder in enumerate(self.sinus)],
            dim=-1,
        )
        timestep_feats = self.timestep_fourier_encoder(timesteps[..., -1])
        timestep_feats = timestep_feats.repeat(1, n_waypoints, 1)
        features = torch.cat((action_feats, timestep_feats), dim=-1)
        features = features.to(self.encoder.trunk[0].weight.dtype)
        encoded = self.encoder(features.flatten(0, 1)).reshape(batch_size, n_waypoints, -1)
        return self.norm(encoded)
