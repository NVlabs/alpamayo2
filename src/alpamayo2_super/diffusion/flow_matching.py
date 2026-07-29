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

"""Flow matching sampler used by the Alpamayo 2 Super expert."""

from typing import Literal

import torch

from alpamayo2_super.diffusion.base import BaseDiffusion, StepFn


class FlowMatching(BaseDiffusion):
    """Flow matching with Euler integration."""

    def __init__(
        self,
        int_method: Literal["euler"] = "euler",
        train_timestep_sampler: Literal["uniform", "beta"] = "beta",
        num_inference_steps: int = 10,
        train_ignore_guidance_rate: float = 0.1,
        inference_guidance_weight: float = 1.0,
        use_classifier_free_guidance: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.int_method = int_method
        self.train_timestep_sampler = train_timestep_sampler
        self.num_inference_steps = num_inference_steps
        self.train_ignore_guidance_rate = train_ignore_guidance_rate
        self.inference_guidance_weight = inference_guidance_weight
        self.use_classifier_free_guidance = use_classifier_free_guidance
        if self.train_timestep_sampler == "beta":
            self.beta_dist = torch.distributions.beta.Beta(
                torch.tensor(1.5, dtype=torch.float32, device="cpu"),
                torch.tensor(1.0, dtype=torch.float32, device="cpu"),
            )
            self.beta_scale_constant = 0.999

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        step_fn: StepFn,
        device: torch.device | str = torch.device("cpu"),
        return_all_steps: bool = False,
        inference_step: int | None = None,
        int_method: Literal["euler"] | None = None,
        *args,
        unguided_step_fn: StepFn | None = None,
        use_classifier_free_guidance: bool | None = None,
        inference_guidance_weight: float | None = None,
        temperature: float = 1.0,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Sample actions by integrating the learned vector field."""
        del args, kwargs
        int_method = int_method or self.int_method
        inference_step = inference_step or self.num_inference_steps
        if use_classifier_free_guidance is None:
            use_classifier_free_guidance = self.use_classifier_free_guidance
        if inference_guidance_weight is None:
            inference_guidance_weight = self.inference_guidance_weight
        if use_classifier_free_guidance and unguided_step_fn is None:
            raise ValueError("unguided_step_fn is required when using classifier free guidance")
        if int_method != "euler":
            raise ValueError(f"Invalid integration method: {int_method}")
        return self._euler(
            batch_size=batch_size,
            step_fn=step_fn,
            unguided_step_fn=unguided_step_fn,
            device=device,
            return_all_steps=return_all_steps,
            inference_step=inference_step,
            inference_guidance_weight=inference_guidance_weight,
            use_classifier_free_guidance=use_classifier_free_guidance,
            temperature=temperature,
        )

    @staticmethod
    def _guided_v(
        step_fn: StepFn,
        x: torch.Tensor,
        t: torch.Tensor,
        unguided_step_fn: StepFn,
        inference_guidance_weight: float,
    ) -> torch.Tensor:
        """Combine conditional and unconditional flow fields."""
        guided_v = step_fn(x=x, t=t)
        unguided_v = unguided_step_fn(x=x, t=t)
        return (1 - inference_guidance_weight) * unguided_v + inference_guidance_weight * guided_v

    def _euler(
        self,
        batch_size: int,
        step_fn: StepFn,
        unguided_step_fn: StepFn | None = None,
        device: torch.device | str = torch.device("cpu"),
        return_all_steps: bool = False,
        inference_step: int | None = None,
        inference_guidance_weight: float = 1.0,
        use_classifier_free_guidance: bool = False,
        temperature: float = 1.0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Integrate the flow field with fixed-step Euler updates."""
        if inference_step is None:
            inference_step = self.num_inference_steps
        if use_classifier_free_guidance and unguided_step_fn is None:
            raise ValueError("unguided_step_fn is required when using classifier free guidance")
        x = torch.randn(batch_size, *self.x_dims, device=device) * temperature
        time_steps = torch.linspace(0.0, 1.0, inference_step + 1, device=device)
        n_dim = len(self.x_dims)
        if return_all_steps:
            all_steps = [x]

        for i in range(inference_step):
            dt = time_steps[i + 1] - time_steps[i]
            dt = dt.view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)
            t_start = time_steps[i].view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)
            if use_classifier_free_guidance:
                v = self._guided_v(
                    step_fn=step_fn,
                    x=x,
                    t=t_start,
                    unguided_step_fn=unguided_step_fn,
                    inference_guidance_weight=inference_guidance_weight,
                )
            else:
                v = step_fn(x=x, t=t_start)
            x = x + dt * v
            if return_all_steps:
                all_steps.append(x)

        if return_all_steps:
            return torch.stack(all_steps, dim=1), time_steps
        return x

    def construct_training_data(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Construct noisy action targets for compatibility with the expert wrapper."""
        batch_size = x.shape[0]

        if self.train_timestep_sampler == "uniform":
            t = torch.rand((batch_size,), device=x.device)
        elif self.train_timestep_sampler == "beta":
            t = self.beta_dist.sample((batch_size,)).to(x.device)
            t = self.beta_scale_constant - t * self.beta_scale_constant
        else:
            raise ValueError(f"Invalid time sampler: {self.train_timestep_sampler}")

        while len(t.shape) < len(x.shape):
            t = t.unsqueeze(-1)

        noise = torch.randn_like(x)
        return {
            "x": x,
            "noisy_x": t * x + (1 - t) * noise,
            "timesteps": t,
            "noise": noise,
            "is_drop_guidance": None,
        }

    def compute_loss_from_pred(
        self,
        training_data: dict[str, torch.Tensor],
        pred: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the flow matching training target loss."""
        x = training_data["x"]
        noise = training_data["noise"]
        target = (x - noise).to(dtype=pred.dtype)
        return torch.nn.functional.mse_loss(target, pred)
