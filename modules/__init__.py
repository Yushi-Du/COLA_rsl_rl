# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural-network components for RL-agents."""

from .actor_critic import ActorCritic
from .actor_critic_end2end import ActorCriticEnd2end
from .actor_critic_end2end_following import ActorCriticEnd2endFollowing
from .actor_critic_wbc_end2end_following import ActorCriticWbcEnd2endFollowing
from .actor_critic_wbc_end2end_quat import ActorCriticWbcEnd2endQuat
from .actor_critic_wbc_end2end_quat_hm_teacher import ActorCriticWbcEnd2endQuatHMTeacher
from .actor_critic_wbc_end2end_quat_transformer import ActorCriticWbcEnd2endQuatTransformer
from .actor_critic_wbc_end2end_following_only_cnn import ActorCriticWbcEnd2endFollowingOnlyCnn
from .actor_critic_wbc_end2end_following_whole_pipe import ActorCriticWbcEnd2endFollowingWholePipe
from .actor_critic_wbc_end2end_following_only_head import ActorCriticWbcEnd2endFollowingOnlyHeadQuat
from .actor_critic_end2end_following_gt_command import ActorCriticEnd2endFollowingGtCommand
from .actor_critic_falcon_wbc_end2end_following import ActorCriticFalconWbcEnd2endFollowing
from .actor_critic_falcon import ActorCriticFalcon
from .actor_critic_wbc_end2end_following_quat import ActorCriticWbcEnd2endFollowingWholePipeQuat
from .actor_critic_wbc_end2end_following_quat_resi import ActorCriticWbcEnd2endFollowingWholePipeQuatResi
from .actor_critic_wbc_end2end_following_quat_resi_vel import ActorCriticWbcEnd2endFollowingWholePipeQuatResiVel
from .actor_critic_wbc_end2end_following_quat_resi_vel_29 import ActorCriticWbcEnd2endFollowingWholePipeQuatResiVel29
from .actor_critic_wbc_end2end_rl_tune import ActorCriticWbcEnd2endRLTuneQuat
from .actor_critic_wbc_end2end_following_quat_resi_vel_15_previ import ActorCriticWbcEnd2endFollowingWholePipeQuatResiVel15Previ
from .actor_critic_wbc_end2end_following_quat_resi_vel_transformer import ActorCriticWbcEnd2endFollowingWholePipeQuatResiVelTransformer
from .actor_critic_wbc_end2end_following_quat_resi_transformer import ActorCriticWbcEnd2endFollowingWholePipeQuatResiTransformer
from .actor_critic_transformer import ActorCriticTransformer
from .actor_critic_recurrent import ActorCriticRecurrent
from .normalizer import EmpiricalNormalization
from .rnd import RandomNetworkDistillation
from .student_teacher import StudentTeacher
from .student_teacher_distill import StudentTeacherDistill
from .student_teacher_distill_hm import StudentTeacherDistill_HM
from .student_teacher_distill_residual import StudentTeacherDistill_Resi
from .student_teacher_recurrent import StudentTeacherRecurrent

__all__ = [
    "ActorCritic",
    "ActorCriticEnd2end",
    "ActorCriticEnd2endFollowing",
    "ActorCriticWbcEnd2endFollowing",
    "ActorCriticWbcEnd2endQuat",
    "ActorCriticWbcEnd2endQuatHMTeacher",
    "ActorCriticWbcEnd2endQuatTransformer",
    "ActorCriticWbcEnd2endFollowingWholePipeQuat",
    "ActorCriticWbcEnd2endFollowingWholePipeQuatResi",
    "ActorCriticWbcEnd2endFollowingWholePipeQuatResiVel",
    "ActorCriticWbcEnd2endFollowingWholePipeQuatResiVel29",
    "ActorCriticWbcEnd2endRLTuneQuat",
    "ActorCriticWbcEnd2endFollowingWholePipeQuatResiVel15Previ",
    "ActorCriticWbcEnd2endFollowingWholePipeQuatResiTransformer",
    "ActorCriticWbcEnd2endFollowingWholePipeQuatResiVelTransformer",
    "ActorCriticEnd2endFollowingGtCommand",
    "ActorCriticFalconWbcEnd2endFollowing",
    "ActorCriticFalcon",
    "ActorCriticWbcEnd2endFollowingWholePipe",
    "ActorCriticWbcEnd2endFollowingOnlyHeadQuat",
    "ActorCriticRecurrent",
    "EmpiricalNormalization",
    "RandomNetworkDistillation",
    "StudentTeacher",
    "StudentTeacherDistill",
    "StudentTeacherDistill_HM",
    "StudentTeacherDistill_Resi",
    "StudentTeacherRecurrent",
]
