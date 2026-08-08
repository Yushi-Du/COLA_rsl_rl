# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runners used by the three-stage COLA training pipeline."""

from .on_policy_runner_end2end import OnPolicyRunnerEnd2end
from .on_policy_runner_whole_pipe_resi import OnPolicyRunnerWholePipeResi

__all__ = ["OnPolicyRunnerEnd2end", "OnPolicyRunnerWholePipeResi"]
