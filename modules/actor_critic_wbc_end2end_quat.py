# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.utils import resolve_nn_activation

import os
from .sensor_cnn import TemporalSensorCNN_Seqlen
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime



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
        new_backbone_logs_dir = os.path.join(os.environ.get("COLA_ROOT", os.getcwd()), "new_backbone_logs")
        base_log_dir = os.path.join(new_backbone_logs_dir, log_time)
        actor_log_dir = os.path.join(base_log_dir, "actor_cnn")
        critic_log_dir = os.path.join(base_log_dir, "critic_cnn")
        os.makedirs(actor_log_dir, exist_ok=True)
        os.makedirs(critic_log_dir, exist_ok=True)
        self.actor_cnn_writer = SummaryWriter(log_dir=actor_log_dir)
        self.critic_cnn_writer = SummaryWriter(log_dir=critic_log_dir)

        self.mono_actor_obs_dim = num_actor_obs
        self.mono_critic_obs_dim = num_critic_obs
        self.actor_cnn = TemporalSensorCNN_Seqlen(in_channels=3, out_channels=32, kernel_size=3, hidden_size=64, output_size=3, seq_len=6)
        self.actor_cnn.train()
        self.actor_cnn_optimizer = torch.optim.Adam(self.actor_cnn.parameters(), lr=1e-4)

        mlp_input_dim_a = self.mono_actor_obs_dim
        mlp_input_dim_c = self.mono_critic_obs_dim
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

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    @staticmethod
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
    
    def actor_cnn_forward(self, observations, inference=False):

        
        commands = observations[:, :, 0:4]
        pose_commands = observations[:, :, 4:18]
        other_features = observations[:, :, 18:]





        return torch.cat([commands, pose_commands, other_features], dim=2)
    
    def critic_cnn_forward(self, observations, inference=False):

        
        commands = observations[:, :, 0:4]
        pose_commands = observations[:, :, 4:18]
        other_features = observations[:, :, 18:]




        return torch.cat([commands, pose_commands, other_features], dim=2)
    
    def process_observations(self, observations, inference=False):
        
        return self.actor(observations)
    
    def process_observations_critic(self, observations, inference=False):

        return self.critic(observations)

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
