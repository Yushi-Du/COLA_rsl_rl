# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from .residual_actor_transformer import TransformerResidualNetwork
from ipdb import set_trace


class StudentTeacherDistill_HM(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
        num_actions,
        student_hidden_dims=[256, 256, 256],
        teacher_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=0.1,
        history_length=10,
        **kwargs,
    ):
        if kwargs:
            print(
                "StudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)
        self.loaded_teacher = False  # indicates if teacher has been loaded
        self.history_length = history_length

        mlp_input_dim_s = num_student_obs
        mlp_input_dim_t = num_teacher_obs
        
        # teacher
        teacher_layers = []
        teacher_layers.append(nn.Linear(mlp_input_dim_t, teacher_hidden_dims[0]))
        teacher_layers.append(activation)
        for layer_index in range(len(teacher_hidden_dims)):
            if layer_index == len(teacher_hidden_dims) - 1:
                teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], num_actions))
            else:
                teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], teacher_hidden_dims[layer_index + 1]))
                teacher_layers.append(activation)
        self.teacher = nn.Sequential(*teacher_layers)
        self.teacher.eval()

        # student
        student_layers = []
        student_layers.append(nn.Linear(mlp_input_dim_s, student_hidden_dims[0]))
        student_layers.append(activation)
        for layer_index in range(len(student_hidden_dims)):
            if layer_index == len(student_hidden_dims) - 1:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], num_actions))
            else:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], student_hidden_dims[layer_index + 1]))
                student_layers.append(activation)
        self.student = nn.Sequential(*student_layers)

        print(f"Student MLP: {self.student}")
        print(f"Teacher: {self.teacher}")

        # action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def reset(self, dones=None, hidden_states=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.student(observations)
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations, inference=None):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        actions_mean = self.student(observations)
        return actions_mean

    def evaluate(self, teacher_observations, inference=None):
        with torch.no_grad():
            actions = self.teacher(teacher_observations)
        return actions

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the student and teacher networks.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters.
        """

        if any("student" in key for key in state_dict.keys()):  # loading parameters from distillation training
            super().load_state_dict(state_dict, strict=strict)
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            return True

        # check if state_dict contains teacher and student or just teacher parameters
        if any("teacher" in key for key in state_dict.keys()):  # loading parameters from rl training
            # rename keys to match teacher and remove critic parameters
            teacher_state_dict = {}
            student_state_dict = {}
            # for key, value in state_dict.items():
            #     if "actor." in key:
            #         teacher_state_dict[key.replace("actor.", "")] = value
            for key, value in state_dict.items():
                if "teacher." in key:
                    teacher_state_dict[key.replace("teacher.", "")] = value
                if "student" in key:
                    student_state_dict[key.replace("student.", "")] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            self.student.load_state_dict(student_state_dict, strict=strict)
            # also load recurrent memory if teacher is recurrent
            if self.is_recurrent and self.teacher_recurrent:
                raise NotImplementedError("Loading recurrent memory for the teacher is not implemented yet")  # TODO
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            if any("student" in key for key in state_dict.keys()):
                return True
            return False
        elif any("actor" in key for key in state_dict.keys()):  # loading parameters from rl training
            # rename keys to match teacher and remove critic parameters
            teacher_state_dict = {}
            # for key, value in state_dict.items():
            #     if "actor." in key:
            #         teacher_state_dict[key.replace("actor.", "")] = value
            for key, value in state_dict.items():
                if "actor." in key and "cnn" not in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            # also load recurrent memory if teacher is recurrent
            if self.is_recurrent and self.teacher_recurrent:
                raise NotImplementedError("Loading recurrent memory for the teacher is not implemented yet")  # TODO
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            return False
        elif any("student" in key for key in state_dict.keys()):  # loading parameters from distillation training
            super().load_state_dict(state_dict, strict=strict)
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            return True
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass


class TeacherResidualWrapper(nn.Module):
    
    def __init__(self, residual_actor, base_actor, history_length):
        super().__init__()
        self.teacher_residual_actor = residual_actor
        self.teacher_base_actor = base_actor
        self.history_length = history_length
    
    def forward(self, observations):
        flattened_obs = observations.reshape(observations.shape[0], self.history_length, -1)
        residual_action = self.teacher_residual_actor(observations)
        
        commands = flattened_obs[:, :, 0:4]
        pose_commands = flattened_obs[:, :, 4:18]
        joint_pos_no_hand = flattened_obs[:, :, 18:18+29]
        joint_vel_no_hand = flattened_obs[:, :, 18+29:18+29+29]
        other_features = flattened_obs[:, :, 18+29+29:-13]
        
        original_commands = torch.cat([commands, pose_commands], dim=2)
        base_actor_obs = torch.cat([original_commands, joint_pos_no_hand, joint_vel_no_hand, other_features], dim=2)
        base_actor_obs_flat = base_actor_obs.reshape(observations.shape[0], -1)
        
        base_action = self.teacher_base_actor(base_actor_obs_flat)
        
        return base_action + residual_action
    
    def eval(self):
        self.teacher_residual_actor.eval()
        self.teacher_base_actor.eval()
        return self
    
    def train(self, mode=True):
        self.teacher_residual_actor.train(mode)
        self.teacher_base_actor.train(mode)
        return self
