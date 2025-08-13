# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

from rsl_rl.utils import resolve_nn_activation

import os
value = os.getenv("IsaacLab_Root")
import sys
sys.path.append(value)
from SensorCNN import SensorCNN, TemporalSensorCNN, TemporalSensorCNN_Seqlen, TemporalSensorCNN_OnlyCnn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from ipdb import set_trace


# class ActorCriticWbcEnd2endFollowingWholePipeQuat(nn.Module):
#     is_recurrent = False

#     def __init__(
#         self,
#         num_actor_obs,
#         num_critic_obs,
#         num_actions,
#         actor_hidden_dims=[256, 256, 256],
#         critic_hidden_dims=[256, 256, 256],
#         activation="elu",
#         init_noise_std=1.0,
#         noise_std_type: str = "scalar",
#         history_length: int = 6,
#         num_envs: int = 2048,
#         device="cuda:0",
#         env=None,
#         **kwargs,
#     ):
#         if kwargs:
#             print(
#                 "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
#                 + str([key for key in kwargs.keys()])
#             )
#         super().__init__()
#         activation = resolve_nn_activation(activation)

#         self.history_length = history_length
#         self.total_steps = 0
#         self.num_envs = num_envs
#         self.device = device
#         self.env = env

#         log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#         new_backbone_logs_dir = os.getenv("LeggedLab_Root") + "/new_backbone_logs"
#         base_log_dir = os.path.join(new_backbone_logs_dir, log_time)
#         actor_log_dir = os.path.join(base_log_dir, "actor_cnn")
#         critic_log_dir = os.path.join(base_log_dir, "critic_cnn")
#         os.makedirs(actor_log_dir, exist_ok=True)
#         os.makedirs(critic_log_dir, exist_ok=True)
#         self.actor_cnn_writer = SummaryWriter(log_dir=actor_log_dir)
#         self.critic_cnn_writer = SummaryWriter(log_dir=critic_log_dir)

#         self.mono_actor_obs_dim = num_actor_obs - int(history_length * 48)
#         self.mono_critic_obs_dim = num_critic_obs - int(history_length * 48)
#         self.actor_cnn = TemporalSensorCNN_OnlyCnn(output_size=18)
#         self.actor_cnn.train()
#         self.actor_cnn_optimizer = torch.optim.Adam(self.actor_cnn.parameters(), lr=1e-3)

#         mlp_input_dim_a = self.mono_actor_obs_dim
#         mlp_input_dim_c = self.mono_critic_obs_dim
#         # Policy
#         actor_layers = []
#         actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
#         actor_layers.append(activation)
#         for layer_index in range(len(actor_hidden_dims)):
#             if layer_index == len(actor_hidden_dims) - 1:
#                 actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
#             else:
#                 actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
#                 actor_layers.append(activation)
#         self.actor = nn.Sequential(*actor_layers)

#         # Value function
#         self.critic_cnn = TemporalSensorCNN_OnlyCnn(output_size=18)
#         self.critic_cnn.train()
#         self.critic_cnn_optimizer = torch.optim.Adam(self.critic_cnn.parameters(), lr=1e-3)

#         critic_layers = []
#         critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
#         critic_layers.append(activation)
#         for layer_index in range(len(critic_hidden_dims)):
#             if layer_index == len(critic_hidden_dims) - 1:
#                 critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
#             else:
#                 critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
#                 critic_layers.append(activation)
#         self.critic = nn.Sequential(*critic_layers)

#         print(f"Actor MLP: {self.actor}")
#         print(f"Critic MLP: {self.critic}")

#         # Action noise
#         self.noise_std_type = noise_std_type
#         if self.noise_std_type == "scalar":
#             self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
#         elif self.noise_std_type == "log":
#             self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
#         else:
#             raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

#         # Action distribution (populated in update_distribution)
#         self.distribution = None
#         # disable args validation for speedup
#         Normal.set_default_validate_args(False)

#     @staticmethod
#     # not used at the moment
#     def init_weights(sequential, scales):
#         [
#             torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
#             for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
#         ]

#     def reset(self, dones=None):
#         pass

#     def forward(self):
#         raise NotImplementedError

#     @property
#     def action_mean(self):
#         return self.distribution.mean

#     @property
#     def action_std(self):
#         return self.distribution.stddev

#     @property
#     def entropy(self):
#         return self.distribution.entropy().sum(dim=-1)
    
#     # 要改
#     def actor_cnn_forward(self, observations, inference=False):

#         commands = observations[:, :, 0:4]
#         pose_commands = observations[:, :, 4:18]
#         tactile_features = observations[:, :, 18:18+48]
#         other_features = observations[:, :, 18+48:]

#         cnn_outputs = self.actor_cnn(tactile_features)  # (num_envs, 16)

#         total_commands = torch.cat([commands, pose_commands], dim=2)  # (num_envs, seq_len, 18)
#         if self.env._actor_cnn_step < self.env.stage_two_steps:
#             if not inference:
#                 self.total_steps += 1
#                 self.actor_cnn_optimizer.zero_grad()
#                 loss = F.mse_loss(cnn_outputs, total_commands)
#                 loss.backward()
#                 self.actor_cnn_optimizer.step()

#                 self.actor_cnn_writer.add_scalar("loss", loss.item(), self.env._actor_cnn_step)
#                 self.env._actor_cnn_step += 1
#                 # print(self.env._actor_cnn_step)
#             final_commands = total_commands  # warmup时短路掉整个actor_cnn
#         else:
#             final_commands = cnn_outputs

#         self.env.predicted_tactile_command = final_commands[:, -1, :]

#         return torch.cat([final_commands, other_features], dim=2)
    
#     def critic_cnn_forward(self, observations, inference=False):

#         commands = observations[:, :, 0:4]
#         pose_commands = observations[:, :, 4:18]
#         tactile_features = observations[:, :, 18:18+48]
#         other_features = observations[:, :, 18+48:]

#         cnn_outputs = self.critic_cnn(tactile_features)  # (num_envs, seq_len, 3)

#         total_commands = torch.cat([commands, pose_commands], dim=2)  # (num_envs, seq_len, 16)
#         if self.env._critic_cnn_step < self.env.stage_two_steps:
#             if not inference:
#                 self.total_steps += 1
#                 self.critic_cnn_optimizer.zero_grad()
#                 loss = F.mse_loss(cnn_outputs, total_commands)
#                 loss.backward()
#                 self.critic_cnn_optimizer.step()

#                 self.critic_cnn_writer.add_scalar("loss", loss.item(), self.env._critic_cnn_step)
#                 self.env._critic_cnn_step += 1
#             final_commands = total_commands  # warmup时短路掉整个critic_cnn
#         else:
#             final_commands = cnn_outputs

#         self.env.predicted_tactile_command = final_commands[:, -1, :]

#         return torch.cat([final_commands, other_features], dim=2)
    
#     def process_observations(self, observations, inference=False):
#         num_envs = observations.shape[0]
#         flattened_obs = observations.reshape(num_envs, self.history_length, -1)
#         mlp_obs_0 = self.actor_cnn_forward(flattened_obs, inference)  
#         total_mlp_obs = mlp_obs_0.reshape(num_envs, -1)
        
#         return self.actor(total_mlp_obs)
    
#     def process_observations_critic(self, observations, inference=False):
#         num_envs = observations.shape[0]
#         flattened_obs = observations.reshape(num_envs, self.history_length, -1)
#         mlp_obs_0 = self.critic_cnn_forward(flattened_obs, inference)  
#         total_mlp_obs = mlp_obs_0.reshape(num_envs, -1)

#         return self.critic(total_mlp_obs)

# 8_13
    # def actor_cnn_forward(self, observations, inference=False):

    #     commands = observations[:, :, 0:4]
    #     pose_commands = observations[:, :, 4:18]
    #     tactile_features = observations[:, :, 18:18+48]
    #     other_features = observations[:, :, 18+48:]

    #     predicted_command = self.command_predictor(tactile_features)  # (num_envs, 16)

    #     total_commands = torch.cat([commands, pose_commands], dim=2)  # (num_envs, seq_len, 18)
    #     if self.env._actor_cnn_step < self.env.stage_two_steps:
    #         if not inference:
    #             self.total_steps += 1
    #             self.actor_cnn_optimizer.zero_grad()
    #             loss = F.mse_loss(cnn_outputs, total_commands)
    #             loss.backward()
    #             self.actor_cnn_optimizer.step()

    #             self.actor_cnn_writer.add_scalar("loss", loss.item(), self.env._actor_cnn_step)
    #             self.env._actor_cnn_step += 1
    #             # print(self.env._actor_cnn_step)
    #         final_commands = total_commands  # warmup时短路掉整个actor_cnn
    #     else:
    #         final_commands = cnn_outputs

    #     self.env.predicted_tactile_command = final_commands[:, -1, :]

    #     return torch.cat([final_commands, other_features], dim=2)
    
    # def critic_cnn_forward(self, observations, inference=False):

    #     commands = observations[:, :, 0:4]
    #     pose_commands = observations[:, :, 4:18]
    #     tactile_features = observations[:, :, 18:18+48]
    #     other_features = observations[:, :, 18+48:]

    #     cnn_outputs = self.critic_cnn(tactile_features)  # (num_envs, seq_len, 3)

    #     total_commands = torch.cat([commands, pose_commands], dim=2)  # (num_envs, seq_len, 16)
    #     if self.env._critic_cnn_step < self.env.stage_two_steps:
    #         if not inference:
    #             self.total_steps += 1
    #             self.critic_cnn_optimizer.zero_grad()
    #             loss = F.mse_loss(cnn_outputs, total_commands)
    #             loss.backward()
    #             self.critic_cnn_optimizer.step()

    #             self.critic_cnn_writer.add_scalar("loss", loss.item(), self.env._critic_cnn_step)
    #             self.env._critic_cnn_step += 1
    #         final_commands = total_commands  # warmup时短路掉整个critic_cnn
    #     else:
    #         final_commands = cnn_outputs

    #     self.env.predicted_tactile_command = final_commands[:, -1, :]

    #     return torch.cat([final_commands, other_features], dim=2)
    
    # def process_observations(self, observations, inference=False):
    #     num_envs = observations.shape[0]
    #     flattened_obs = observations.reshape(num_envs, self.history_length, -1)
    #     mlp_obs_0 = self.actor_cnn_forward(flattened_obs, inference)  
    #     total_mlp_obs = mlp_obs_0.reshape(num_envs, -1)
        
    #     return self.actor(total_mlp_obs)
    
    # def process_observations_critic(self, observations, inference=False):
    #     num_envs = observations.shape[0]
    #     flattened_obs = observations.reshape(num_envs, self.history_length, -1)
    #     mlp_obs_0 = self.critic_cnn_forward(flattened_obs, inference)  
    #     total_mlp_obs = mlp_obs_0.reshape(num_envs, -1)

    #     return self.critic(total_mlp_obs)
# /8_13

#     def update_distribution(self, observations, inference=False):

#         mean = self.process_observations(observations, inference)
#         # print(mean.shape)
#         # if mean.shape[0] != self.num_envs:
#         #     set_trace()
#         # compute standard deviation
#         if self.noise_std_type == "scalar":
#             std = self.std.expand_as(mean)
#         elif self.noise_std_type == "log":
#             std = torch.exp(self.log_std).expand_as(mean)
#         else:
#             raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

#         # # 额外的安全检查和调试信息
#         # if torch.any(torch.isnan(std)) or torch.any(std <= 0):
#         #     print(f"WARNING: Still found invalid std values, using default std=1.0")
#         #     std = torch.ones_like(std)

#         # create distribution
#         self.distribution = Normal(mean, std)

#     def act(self, observations, inference=False, **kwargs):
#         self.update_distribution(observations, inference)
#         return self.distribution.sample()

#     def get_actions_log_prob(self, actions):
#         return self.distribution.log_prob(actions).sum(dim=-1)

#     def act_inference(self, observations):
#         actions_mean = self.process_observations(observations, inference=True)
#         return actions_mean

#     def evaluate(self, critic_observations, inference=False, **kwargs):
#         # value = self.critic(critic_observations)
#         value = self.process_observations_critic(critic_observations, inference)
#         return value

#     def load_state_dict(self, state_dict, strict=True):
#         """Load the parameters of the actor-critic model.

#         Args:
#             state_dict (dict): State dictionary of the model.
#             strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
#                            module's state_dict() function.

#         Returns:
#             bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
#                   `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
#         """

#         super().load_state_dict(state_dict, strict=False)
#         return True


class ActorCriticWbcEnd2endFollowingWholePipeQuat(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        history_length: int = 10,
        num_envs: int = 2048,
        device="cuda:0",
        env=None,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        self.history_length = history_length
        self.total_steps = 0
        self.num_envs = num_envs
        self.device = device
        self.env = env
        self.predicted_command = None
        self.num_commands = 18

        log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_backbone_logs_dir = os.getenv("LeggedLab_Root") + "/new_backbone_logs"
        base_log_dir = os.path.join(new_backbone_logs_dir, log_time)
        command_predictor_log_dir = os.path.join(base_log_dir, "command_predictor")
        os.makedirs(command_predictor_log_dir, exist_ok=True)
        self.command_predictor_writer = SummaryWriter(log_dir=command_predictor_log_dir)

        self.actor_command_predictor_obs_dim = num_actor_obs

        self.actor_obs_dim = num_actor_obs - self.history_length * self.num_commands
        self.critic_obs_dim = num_critic_obs - self.history_length * self.num_commands

        # Command predictor
        actor_command_predictor_layers = []
        actor_command_predictor_layers.append(nn.Linear(self.actor_command_predictor_obs_dim, actor_hidden_dims[0]))
        actor_command_predictor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_command_predictor_layers.append(nn.Linear(actor_hidden_dims[layer_index], self.num_commands))
            else:
                actor_command_predictor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_command_predictor_layers.append(activation)
        self.actor_command_predictor = nn.Sequential(*actor_command_predictor_layers)
        self.actor_command_predictor_optimizer = torch.optim.Adam(self.actor_command_predictor.parameters(), lr=1e-3)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(self.actor_obs_dim, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(self.critic_obs_dim, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # New Value function for predicted commands
        critic_for_predicted_commands_layers = []
        critic_for_predicted_commands_layers.append(nn.Linear(self.critic_obs_dim, critic_hidden_dims[0]))
        critic_for_predicted_commands_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_for_predicted_commands_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_for_predicted_commands_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_for_predicted_commands_layers.append(activation)
        self.critic_for_predicted_commands = nn.Sequential(*critic_for_predicted_commands_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"New Critic MLP: {self.critic_for_predicted_commands}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions+self.num_commands))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions+self.num_commands)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
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

    def actor_forward(self, observations, inference=False):
        # observations: (-1, history_length*single_frame_observations)
        flattened_obs = observations.reshape(observations.shape[0], self.history_length, -1)

        commands = flattened_obs[:, :, 0:4]
        pose_commands = flattened_obs[:, :, 4:18]
        other_features = flattened_obs[:, :, 18:-18]  # (-1, history_length, other_features_dim)
        predicted_commands_history = flattened_obs[:, :, -18:]  # (-1, history_length, 18)

        predicted_command = self.actor_command_predictor(observations)  # (-1, 18)

        original_commands = torch.cat([commands, pose_commands], dim=2)  # (-1, history_length, 18)
        if self.env._actor_cnn_step < self.env.stage_two_steps:
            if not inference:
                self.total_steps += 1
                self.actor_command_predictor_optimizer.zero_grad()
                loss = F.mse_loss(predicted_command, original_commands[:, -1, :])
                loss.backward()
                self.actor_command_predictor_optimizer.step()

                self.command_predictor_writer.add_scalar("loss", loss.item(), self.env._actor_cnn_step)
                self.env._actor_cnn_step += 1

                predicted_command = predicted_command.detach()

            final_commands = original_commands  # command_predictor
        else:
            final_commands = predicted_commands_history

        try:
            actor_observations = torch.cat([final_commands, other_features], dim=2).reshape(observations.shape[0], -1)
        except:
            import tracebackW
            print("SHAPE MISMATCH DETECTED!")
            print("Call stack:")
            traceback.print_stack()
            set_trace()
        # if self.env._actor_cnn_step < self.env.stage_two_steps:
        #     assert torch.equal(actor_observations[:, 111:222], observations[:, 129:240])

        return actor_observations, predicted_command
    
    # 第一种方式：老的critics还是给29维的action打分，只不过新增了一个根据previleged信息生成更真实predicted_command的网络，不太合理因为不知道新增的预测command的网络该怎样用
    def critic_forward(self, observations, inference=False):
        # observations: (-1, history_length*single_frame_observations)
        flattened_obs = observations.reshape(observations.shape[0], self.history_length, -1)

        commands = flattened_obs[:, :, 0:4]
        pose_commands = flattened_obs[:, :, 4:18]
        other_features = flattened_obs[:, :, 18:18+3*2+29*3]  # (-1, history_length, other_features_dim)
        predicted_commands_history = flattened_obs[:, :, 18+3*2+29*3:18+3*2+29*3+18]  # (-1, history_length, 18)
        other_critic_features = flattened_obs[:, :, 18+3*2+29*3+18:]

        original_commands = torch.cat([commands, pose_commands], dim=2)  # (-1, history_length, 18)
        if self.env._actor_cnn_step < self.env.stage_two_steps:
            new_critic_score = 0
        else:
            new_critic_score = self.critic_for_predicted_commands(observations)  # (-1, 1)

        try:
            critic_observations = torch.cat([original_commands, other_features, other_critic_features], dim=2).reshape(observations.shape[0], -1)
        except:
            import traceback
            print("SHAPE MISMATCH DETECTED!")
            print("Call stack:")
            traceback.print_stack()
            set_trace()
        
        # if self.env._actor_cnn_step < self.env.stage_two_steps:
        #     assert torch.equal(critic_observations[:, 116:227], observations[:, 134:245])
        return critic_observations, new_critic_score
    
    def process_observations(self, observations, inference=False):
        total_mlp_obs, predicted_command = self.actor_forward(observations, inference)
        return torch.cat([self.actor(total_mlp_obs), predicted_command], dim=-1)
    
    def process_observations_critic(self, observations, inference=False):
        total_mlp_obs, new_critic_score = self.critic_forward(observations, inference)
        return self.critic(total_mlp_obs)+new_critic_score

    def update_distribution(self, observations, inference=False):

        mean = self.process_observations(observations, inference)

        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = Normal(mean, std)

    def act(self, observations, inference=False, **kwargs):
        self.update_distribution(observations, inference)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.process_observations(observations, inference=True)
        return actions_mean

    def evaluate(self, critic_observations, inference=False, **kwargs):
        # value = self.critic(critic_observations)
        value = self.process_observations_critic(critic_observations, inference)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        """处理参数维度扩展的加载"""
    
        current_state = self.state_dict()
        
        # 处理 std 参数的维度扩展
        if 'std' in state_dict and 'std' in current_state:
            saved_std = state_dict['std']
            current_std = current_state['std']
            
            if saved_std.shape[0] < current_std.shape[0]:
                print(f"扩展 std 参数: {saved_std.shape[0]} → {current_std.shape[0]}")
                # 保持当前初始化值，只更新前29维
                extended_std = current_std.clone()
                extended_std[:saved_std.shape[0]] = saved_std
                state_dict['std'] = extended_std
                print(f"前 {saved_std.shape[0]} 维使用检查点值，后 {current_std.shape[0] - saved_std.shape[0]} 维保持初始化值")
        
        # 处理 log_std 参数（如果存在）
        if 'log_std' in state_dict and 'log_std' in current_state:
            saved_log_std = state_dict['log_std']
            current_log_std = current_state['log_std']
            
            if saved_log_std.shape[0] < current_log_std.shape[0]:
                print(f"扩展 log_std 参数: {saved_log_std.shape[0]} → {current_log_std.shape[0]}")
                extended_log_std = current_log_std.clone()
                extended_log_std[:saved_log_std.shape[0]] = saved_log_std
                state_dict['log_std'] = extended_log_std
        
        # 现在可以安全加载了
        super().load_state_dict(state_dict, strict=False)
        return True
