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
from SensorCNN import SensorCNN, TemporalSensorCNN, TemporalSensorCNN_Seqlen
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from ipdb import set_trace


class ActorCriticWbcEnd2endQuat(nn.Module):
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
        history_length: int = 6,
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

        log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_backbone_logs_dir = os.getenv("LeggedLab_Root") + "/new_backbone_logs"
        base_log_dir = os.path.join(new_backbone_logs_dir, log_time)
        actor_log_dir = os.path.join(base_log_dir, "actor_cnn")
        critic_log_dir = os.path.join(base_log_dir, "critic_cnn")
        os.makedirs(actor_log_dir, exist_ok=True)
        os.makedirs(critic_log_dir, exist_ok=True)
        self.actor_cnn_writer = SummaryWriter(log_dir=actor_log_dir)
        self.critic_cnn_writer = SummaryWriter(log_dir=critic_log_dir)

        self.mono_actor_obs_dim = num_actor_obs - int(history_length * 48)
        self.mono_critic_obs_dim = num_critic_obs - int(history_length * 48)
        self.actor_cnn = TemporalSensorCNN_Seqlen(in_channels=3, out_channels=32, kernel_size=3, hidden_size=64, output_size=3, seq_len=6)
        self.actor_cnn.train()
        self.actor_cnn_optimizer = torch.optim.Adam(self.actor_cnn.parameters(), lr=1e-4)

        mlp_input_dim_a = self.mono_actor_obs_dim
        mlp_input_dim_c = self.mono_critic_obs_dim
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        self.critic_cnn = TemporalSensorCNN_Seqlen(in_channels=3, out_channels=32, kernel_size=3, hidden_size=64, output_size=3, seq_len=6)
        self.critic_cnn.train()
        self.critic_cnn_optimizer = torch.optim.Adam(self.critic_cnn.parameters(), lr=1e-4)

        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
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
    
    # 要改
    def actor_cnn_forward(self, observations, inference=False):
        # observations.shape: (num_envs, history_length, 240)
        # commands = observations[:, :, 0:3]
        # pose_commands = observations[:, :, 3:15]
        # # set_trace()
        # tactile_features = observations[:, :, 15:15+144]
        # other_features = observations[:, :, 15+144:]

        commands = observations[:, :, 0:4]
        pose_commands = observations[:, :, 4:18]
        tactile_features = observations[:, :, 18:18+48]
        other_features = observations[:, :, 18+48:]

        # recovered_outputs = tactile_features.reshape(tactile_features.shape[0], tactile_features.shape[1], 48, 3)
        # cnn_outputs = self.actor_cnn(recovered_outputs)  # (num_envs, seq_len, 3)

        # if self.env._actor_cnn_step < self.env.stage_two_steps:
        #     # print('Here!')
        #     if not inference:
        #         self.total_steps += 1
        #         self.actor_cnn_optimizer.zero_grad()
        #         loss = F.mse_loss(cnn_outputs, commands)
        #         loss.backward()
        #         self.actor_cnn_optimizer.step()

        #         self.actor_cnn_writer.add_scalar("loss", loss.item(), self.env._actor_cnn_step)
        #         self.env._actor_cnn_step += 1
        #         # print(self.env._actor_cnn_step)
        #     final_commands = commands  # warmup时短路掉整个actor_cnn
        # else:
        #     final_commands = cnn_outputs

        # self.env.predicted_tactile_command = final_commands[:, -1, :]

        # return torch.cat([final_commands, other_features], dim=2)
        return torch.cat([commands, pose_commands, other_features], dim=2)
    
    def critic_cnn_forward(self, observations, inference=False):
        # observations.shape: (num_envs, history_length, 240)
        # commands = observations[:, :, 0:3]
        # pose_commands = observations[:, :, 3:15]
        # tactile_features = observations[:, :, 15:15+144]
        # other_features = observations[:, :, 15+144:]

        commands = observations[:, :, 0:4]
        pose_commands = observations[:, :, 4:18]
        tactile_features = observations[:, :, 18:18+48]
        other_features = observations[:, :, 18+48:]

        # recovered_outputs = tactile_features.reshape(tactile_features.shape[0], tactile_features.shape[1], 48, 3)
        # cnn_outputs = self.critic_cnn(recovered_outputs)  # (num_envs, seq_len, 3)

        # final_commands = cnn_outputs
        # if self.env._critic_cnn_step < self.env.stage_two_steps:
        #     if not inference:
        #         self.total_steps += 1
        #         self.critic_cnn_optimizer.zero_grad()
        #         loss = F.mse_loss(cnn_outputs, commands)
        #         loss.backward()
        #         self.critic_cnn_optimizer.step()

        #         self.critic_cnn_writer.add_scalar("loss", loss.item(), self.env._critic_cnn_step)
        #         self.env._critic_cnn_step += 1
        #     final_commands = commands  # warmup时短路掉整个actor_cnn

        # return torch.cat([final_commands, other_features], dim=2)
        return torch.cat([commands, pose_commands, other_features], dim=2)
    
    def process_observations(self, observations, inference=False):
        num_envs = observations.shape[0]
        flattened_obs = observations.reshape(num_envs, self.history_length, -1)
        mlp_obs_0 = self.actor_cnn_forward(flattened_obs, inference)  
        total_mlp_obs = mlp_obs_0.reshape(num_envs, -1)
        
        return self.actor(total_mlp_obs)
    
    def process_observations_critic(self, observations, inference=False):
        num_envs = observations.shape[0]
        flattened_obs = observations.reshape(num_envs, self.history_length, -1)
        mlp_obs_0 = self.critic_cnn_forward(flattened_obs, inference)  
        total_mlp_obs = mlp_obs_0.reshape(num_envs, -1)

        return self.critic(total_mlp_obs)

    def update_distribution(self, observations, inference=False):

        mean = self.process_observations(observations, inference)
        # print(mean.shape)
        # if mean.shape[0] != self.num_envs:
        #     set_trace()
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
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

        super().load_state_dict(state_dict, strict=strict)
        return True
