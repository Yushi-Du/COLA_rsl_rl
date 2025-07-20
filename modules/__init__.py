# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural-network components for RL-agents."""

from .actor_critic import ActorCritic
from .actor_critic_end2end import ActorCriticEnd2end
from .actor_critic_end2end_following import ActorCriticEnd2endFollowing
from .actor_critic_wbc_end2end_following import ActorCriticWbcEnd2endFollowing
from .actor_critic_wbc_end2end_following_only_cnn import ActorCriticWbcEnd2endFollowingOnlyCnn
from .actor_critic_end2end_following_gt_command import ActorCriticEnd2endFollowingGtCommand
from .actor_critic_falcon_wbc_end2end_following import ActorCriticFalconWbcEnd2endFollowing
from .actor_critic_transformer import ActorCriticTransformer
from .actor_critic_recurrent import ActorCriticRecurrent
from .normalizer import EmpiricalNormalization
from .rnd import RandomNetworkDistillation
from .student_teacher import StudentTeacher
from .student_teacher_recurrent import StudentTeacherRecurrent

__all__ = [
    "ActorCritic",
    "ActorCriticEnd2end",
    "ActorCriticEnd2endFollowing",
    "ActorCriticWbcEnd2endFollowing",
    "ActorCriticEnd2endFollowingGtCommand",
    "ActorCriticFalconWbcEnd2endFollowing",
    "ActorCriticRecurrent",
    "EmpiricalNormalization",
    "RandomNetworkDistillation",
    "StudentTeacher",
    "StudentTeacherRecurrent",
]
