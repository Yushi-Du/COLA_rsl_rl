# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural-network components used by the COLA training pipeline."""

from .actor_critic_wbc_end2end_quat import ActorCriticWbcEnd2endQuat
from .actor_critic_wbc_end2end_following_quat_resi_vel_29 import (
    ActorCriticWbcEnd2endFollowingWholePipeQuatResiVel29,
)
from .normalizer import EmpiricalNormalization
from .rnd import RandomNetworkDistillation
from .student_teacher_distill import StudentTeacherDistill

__all__ = [
    "ActorCriticWbcEnd2endQuat",
    "ActorCriticWbcEnd2endFollowingWholePipeQuatResiVel29",
    "EmpiricalNormalization",
    "RandomNetworkDistillation",
    "StudentTeacherDistill",
]
