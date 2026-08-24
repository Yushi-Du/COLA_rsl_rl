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
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class ActorCriticWbcEnd2endFollowingWholePipeQuatResiVel29(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        residual_actor_hidden_dims=None,
        residual_critic_hidden_dims=None,
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        history_length: int = 10,
        num_envs: int = 2048,
        residual_hidden_init_std: float | None = None,
        residual_final_init_std: float | None = None,
        residual_bias_init: float | None = None,
        base_privileged_obs_per_frame: int = 13,
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
        if None in (
            residual_hidden_init_std,
            residual_final_init_std,
            residual_bias_init,
        ):
            raise ValueError("Residual initialization values must be provided by the task config")
        activation = resolve_nn_activation(activation)

        self.history_length = history_length
        self.total_steps = 0
        self.num_envs = num_envs
        self.device = device
        self.env = env
        self.predicted_command = None
        self.residual_hidden_init_std = residual_hidden_init_std
        self.residual_final_init_std = residual_final_init_std
        self.residual_bias_init = residual_bias_init
        residual_actor_hidden_dims = (
            actor_hidden_dims
            if residual_actor_hidden_dims is None
            else residual_actor_hidden_dims
        )
        residual_critic_hidden_dims = (
            critic_hidden_dims
            if residual_critic_hidden_dims is None
            else residual_critic_hidden_dims
        )
        if not residual_actor_hidden_dims or not residual_critic_hidden_dims:
            raise ValueError("Residual hidden-dimension lists must not be empty")
        if any(width <= 0 for width in residual_actor_hidden_dims):
            raise ValueError("Residual actor hidden dimensions must be positive")
        if any(width <= 0 for width in residual_critic_hidden_dims):
            raise ValueError("Residual critic hidden dimensions must be positive")
        if base_privileged_obs_per_frame < 0:
            raise ValueError("base_privileged_obs_per_frame must be non-negative")
        self.base_privileged_obs_per_frame = base_privileged_obs_per_frame

        log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_backbone_logs_dir = os.path.join(os.environ.get("COLA_ROOT", os.getcwd()), "new_backbone_logs")
        base_log_dir = os.path.join(new_backbone_logs_dir, log_time)
        residual_log_dir = os.path.join(base_log_dir, "residual")
        os.makedirs(residual_log_dir, exist_ok=True)
        self.residual_writer = SummaryWriter(log_dir=residual_log_dir)

        self.residual_actor_obs_dim = num_actor_obs
        self.residual_critic_obs_dim = num_critic_obs

        self.actor_obs_dim = (
            num_actor_obs
            - self.history_length * self.base_privileged_obs_per_frame
        )
        self.critic_obs_dim = (
            num_critic_obs
            - self.history_length * self.base_privileged_obs_per_frame
        )
        if self.actor_obs_dim <= 0 or self.critic_obs_dim <= 0:
            raise ValueError(
                "base privileged observation tail is larger than the observation"
            )

        self.command_idx = 4
        self.pose_command_idx = 4+14
        self.ang_vel_idx = 4+14+3
        self.projected_gravity_idx = 4+14+3+3
        self.joint_pos_idx = 4+14+3+3+29
        self.joint_vel_idx = 4+14+3+3+29+29
        self.prev_action_idx = 4+14+3+3+29+29+29

        residual_actor_layers = []
        residual_actor_layers.append(
            nn.Linear(self.residual_actor_obs_dim, residual_actor_hidden_dims[0])
        )
        residual_actor_layers.append(activation)
        for layer_index in range(len(residual_actor_hidden_dims)):
            if layer_index == len(residual_actor_hidden_dims) - 1:
                residual_actor_layers.append(
                    nn.Linear(residual_actor_hidden_dims[layer_index], num_actions)
                )
            else:
                residual_actor_layers.append(
                    nn.Linear(
                        residual_actor_hidden_dims[layer_index],
                        residual_actor_hidden_dims[layer_index + 1],
                    )
                )
                residual_actor_layers.append(activation)
        self.residual_actor = nn.Sequential(*residual_actor_layers)

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

        residual_critic_layers = []
        residual_critic_layers.append(
            nn.Linear(self.residual_critic_obs_dim, residual_critic_hidden_dims[0])
        )
        residual_critic_layers.append(activation)
        for layer_index in range(len(residual_critic_hidden_dims)):
            if layer_index == len(residual_critic_hidden_dims) - 1:
                residual_critic_layers.append(
                    nn.Linear(residual_critic_hidden_dims[layer_index], 1)
                )
            else:
                residual_critic_layers.append(
                    nn.Linear(
                        residual_critic_hidden_dims[layer_index],
                        residual_critic_hidden_dims[layer_index + 1],
                    )
                )
                residual_critic_layers.append(activation)
        self.residual_critic = nn.Sequential(*residual_critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Residual Actor MLP: {self.residual_actor}")
        print(f"Residual Critic MLP: {self.residual_critic}")

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

        self._freeze_base_networks()

        self._initialize_residual_networks()
    
    def _freeze_base_networks(self):
        """Freeze the base actor and critic during residual training."""
        for param in self.actor.parameters():
            param.requires_grad = False
        for param in self.critic.parameters():
            param.requires_grad = False
        
        print("Frozen base-network parameters:")
        print(f"  Actor: {sum(p.numel() for p in self.actor.parameters()):,} parameters")
        print(f"  Critic: {sum(p.numel() for p in self.critic.parameters()):,} parameters")
        print("Trainable residual parameters:")
        print(f"  Residual Actor: {sum(p.numel() for p in self.residual_actor.parameters()):,} parameters")
        print(f"  Residual Critic: {sum(p.numel() for p in self.residual_critic.parameters()):,} parameters")

    def _initialize_residual_networks(self):
        """Initialize the residual actor and critic near zero."""
        
        def init_layer(layer, is_final=False):
            if isinstance(layer, nn.Linear):
                if is_final:
                    nn.init.normal_(
                        layer.weight,
                        mean=0.0,
                        std=self.residual_final_init_std,
                    )
                    nn.init.constant_(layer.bias, self.residual_bias_init)
                    print(f"  Initialized final layer near zero: {layer}")
                else:
                    nn.init.normal_(
                        layer.weight,
                        mean=0.0,
                        std=self.residual_hidden_init_std,
                    )
                    nn.init.constant_(layer.bias, self.residual_bias_init)
        
        print("Initializing residual networks:")
        
        print("- Residual Actor:")
        for i, layer in enumerate(self.residual_actor):
            is_final = (i == len(self.residual_actor) - 1)
            init_layer(layer, is_final)
        
        print("- Residual Critic:")
        for i, layer in enumerate(self.residual_critic):
            is_final = (i == len(self.residual_critic) - 1)
            init_layer(layer, is_final)
        
        self._verify_initialization()
    
    def _verify_initialization(self):
        """Report the initial residual-network output ranges."""
        with torch.no_grad():
            dummy_obs = torch.randn(32, self.residual_actor_obs_dim, device=next(self.residual_actor.parameters()).device)
            dummy_obs_critic = torch.randn(32, self.residual_critic_obs_dim, device=next(self.residual_critic.parameters()).device)
            
            residual_actor_out = self.residual_actor(dummy_obs)
            residual_critic_out = self.residual_critic(dummy_obs_critic)
            
            print("Initialization verification:")
            print(f"  Residual Actor range: [{residual_actor_out.min():.6f}, {residual_actor_out.max():.6f}]")
            print(f"  Residual Actor mean absolute value: {residual_actor_out.abs().mean():.6f}")
            print(f"  Residual Critic range: [{residual_critic_out.min():.6f}, {residual_critic_out.max():.6f}]")
            print(f"  Residual Critic mean absolute value: {residual_critic_out.abs().mean():.6f}")
            
            if residual_actor_out.abs().mean() < 1e-5:
                print("  Residual Actor was initialized near zero")
            else:
                print("  Warning: Residual Actor initialization may be incorrect")
                
            if residual_critic_out.abs().mean() < 1e-5:
                print("  Residual Critic was initialized near zero")
            else:
                print("  Warning: Residual Critic initialization may be incorrect")

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

    def actor_forward(self, observations, inference=False):
        # observations: (-1, history_length*single_frame_observations)
        flattened_obs = observations.reshape(observations.shape[0], self.history_length, -1)

        commands = flattened_obs[:, :, 0:4]
        pose_commands = flattened_obs[:, :, 4:18]
        joint_pos = flattened_obs[:, :, 18:18+29]  # (-1, history_length, 29)
        joint_vel = flattened_obs[:, :, 18+29:18+29+29]  # (-1, history_length, 29)
        privileged_start = (
            -self.base_privileged_obs_per_frame
            if self.base_privileged_obs_per_frame
            else None
        )
        other_features = flattened_obs[:, :, 18+29+29:privileged_start]
        
        self.total_steps += 1

        residual_action = self.residual_actor(observations)  # (-1, 29)

        original_commands = torch.cat([commands, pose_commands], dim=2)  # (-1, history_length, 18)
        actor_observations = torch.cat([original_commands, joint_pos, joint_vel, other_features], dim=2).reshape(observations.shape[0], -1)

        return actor_observations, residual_action
    
    def critic_forward(self, observations, inference=False):
        # observations: (-1, history_length*single_frame_observations)
        flattened_obs = observations.reshape(observations.shape[0], self.history_length, -1)

        commands = flattened_obs[:, :, 0:4]
        pose_commands = flattened_obs[:, :, 4:18]
        joint_pos = flattened_obs[:, :, 18:18+29]  # (-1, history_length, 29)
        joint_vel = flattened_obs[:, :, 18+29:18+29+29]  # (-1, history_length, 29)
        privileged_start = -(self.base_privileged_obs_per_frame + 5)
        other_features = flattened_obs[:, :, 18+29+29:privileged_start]
        other_critic_features = flattened_obs[:, :, -5:]

        original_commands = torch.cat([commands, pose_commands], dim=2)  # (-1, history_length, 18)
        residual_critic_score = self.residual_critic(observations)  # (-1, 1)
        critic_observations = torch.cat([original_commands, joint_pos, joint_vel, other_features, other_critic_features], dim=2).reshape(observations.shape[0], -1)
        
        return critic_observations, residual_critic_score
    
    def process_observations(self, observations, inference=False):
        total_mlp_obs, residual_action = self.actor_forward(observations, inference)
        ori_action = self.actor(total_mlp_obs)
        self.residual_writer.add_scalar("residual_action_ratio", (residual_action/ori_action).mean(), self.total_steps)
        return ori_action + residual_action
    
    def process_observations_critic(self, observations, inference=False):
        total_mlp_obs, residual_critic_score = self.critic_forward(observations, inference)
        ori_critic_score = self.critic(total_mlp_obs)
        self.residual_writer.add_scalar("residual_critic_ratio", (residual_critic_score/ori_critic_score).mean(), self.total_steps)
        return ori_critic_score+residual_critic_score

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

        is_same_phase = any(
            key.startswith("residual_actor.")
            or key.startswith("residual_critic.")
            for key in state_dict
        )
        super().load_state_dict(
            state_dict,
            strict=strict if is_same_phase else False,
        )
        # DDP broadcast_parameters re-enters this on every distributed init: the
        # source tensor is on rank 0's cuda:0 while self.actor sits on cuda:rank.
        # torch.equal across devices raises, so move both to the same device.
        _self_bias = self.actor.state_dict()['0.bias']
        assert torch.equal(state_dict['actor.0.bias'].to(_self_bias.device), _self_bias)
        return is_same_phase
