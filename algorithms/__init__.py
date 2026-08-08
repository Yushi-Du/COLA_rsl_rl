# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Algorithms used by the three-stage COLA training pipeline."""

from .distillation_distill import DistillationDistill
from .ppo_wbc_end2end_quat import PPO_WbcEnd2endQuat
from .ppo_wbc_end2end_whole_pipe_resi_vel import PPO_WbcEnd2endWholePipeResiVel

__all__ = [
    "PPO_WbcEnd2endQuat",
    "PPO_WbcEnd2endWholePipeResiVel",
    "DistillationDistill",
]
