# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

from .distillation import Distillation
from .distillation_distill import DistillationDistill
from .ppo import PPO
from .ppo_end2end import PPO_End2end
from .ppo_wbc_end2end import PPO_WbcEnd2end
from .ppo_wbc_end2end_quat import PPO_WbcEnd2endQuat
from .ppo_wbc_end2end_only_cnn import PPO_WbcEnd2endOnlyCnn
from .ppo_wbc_end2end_whole_pipe import PPO_WbcEnd2endWholePipe
from .ppo_wbc_end2end_whole_pipe_resi import PPO_WbcEnd2endWholePipeResi
from .ppo_wbc_end2end_only_head import PPO_WbcEnd2endOnlyHead
from .ppo_end2end_gt_command import PPO_End2endGtCommand
from .ppo_end2end_falcon_wbc_end2end_following import PPO_FalconWbcEnd2endFollowing
from .ppo_falcon import PPO_Falcon

__all__ = ["PPO", "PPO_End2end", "PPO_WbcEnd2end", "PPO_WbcEnd2endQuat", "PPO_WbcEnd2endOnlyCnn", "PPO_WbcEnd2endOnlyHead", "PPO_WbcEnd2endWholePipe", "PPO_WbcEnd2endWholePipeResi", "PPO_End2endGtCommand", "PPO_FalconWbcEnd2endFollowing", "PPO_Falcon", "Distillation", "DistillationDistill"]
