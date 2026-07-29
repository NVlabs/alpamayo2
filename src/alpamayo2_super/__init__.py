# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alpamayo2_super.config import Alpamayo2SuperConfig
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

__all__ = ["Alpamayo2Super", "Alpamayo2SuperConfig"]


def __getattr__(name: str):
    """Lazily expose public classes without importing heavy model dependencies."""
    if name == "Alpamayo2SuperConfig":
        from alpamayo2_super.config import Alpamayo2SuperConfig

        return Alpamayo2SuperConfig
    if name == "Alpamayo2Super":
        from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

        return Alpamayo2Super
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
