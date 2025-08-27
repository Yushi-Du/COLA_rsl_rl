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


class ActorCriticWbcEnd2endFollowingWholePipeQuatResiVel15Previ(nn.Module):
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

        log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_backbone_logs_dir = os.getenv("LeggedLab_Root") + "/new_backbone_logs"
        base_log_dir = os.path.join(new_backbone_logs_dir, log_time)
        residual_log_dir = os.path.join(base_log_dir, "residual")
        os.makedirs(residual_log_dir, exist_ok=True)
        self.residual_writer = SummaryWriter(log_dir=residual_log_dir)

        self.residual_actor_obs_dim = num_actor_obs
        self.residual_critic_obs_dim = num_critic_obs

        self.actor_obs_dim = num_actor_obs - self.history_length * (15)
        self.critic_obs_dim = num_critic_obs - self.history_length * (15)

        self.command_idx = 4
        self.pose_command_idx = 4+14
        self.ang_vel_idx = 4+14+3
        self.projected_gravity_idx = 4+14+3+3
        self.joint_pos_idx = 4+14+3+3+29
        self.joint_vel_idx = 4+14+3+3+29+29
        self.prev_action_idx = 4+14+3+3+29+29+29

        # Residual Actor
        residual_actor_layers = []
        residual_actor_layers.append(nn.Linear(self.residual_actor_obs_dim, actor_hidden_dims[0]))
        residual_actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                residual_actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                residual_actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                residual_actor_layers.append(activation)
        self.residual_actor = nn.Sequential(*residual_actor_layers)

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
        residual_critic_layers = []
        residual_critic_layers.append(nn.Linear(self.residual_critic_obs_dim, critic_hidden_dims[0]))
        residual_critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                residual_critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                residual_critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                residual_critic_layers.append(activation)
        self.residual_critic = nn.Sequential(*residual_critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Residual Actor MLP: {self.residual_actor}")
        print(f"Residual Critic MLP: {self.residual_critic}")

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

        self._freeze_base_networks()

        self._initialize_residual_networks()
    
    def _freeze_base_networks(self):
        """冻结原网络参数，提高训练效率"""
        for param in self.actor.parameters():
            param.requires_grad = False
        for param in self.critic.parameters():
            param.requires_grad = False
        
        print(f"冻结原网络参数:")
        print(f"  Actor: {sum(p.numel() for p in self.actor.parameters()):,} 参数")
        print(f"  Critic: {sum(p.numel() for p in self.critic.parameters()):,} 参数")
        print(f"可训练 Residual 参数:")
        print(f"  Residual Actor: {sum(p.numel() for p in self.residual_actor.parameters()):,} 参数")
        print(f"  Residual Critic: {sum(p.numel() for p in self.residual_critic.parameters()):,} 参数")

    def _initialize_residual_networks(self):
        """零初始化 residual 网络"""
        
        def init_layer(layer, is_final=False):
            if isinstance(layer, nn.Linear):
                if is_final:
                    # 最后一层：完全零初始化
                    nn.init.zeros_(layer.weight)
                    nn.init.zeros_(layer.bias)
                    print(f"  零初始化最后层: {layer}")
                else:
                    # 中间层：小随机初始化
                    nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(layer.bias)
        
        print("初始化 Residual 网络:")
        
        # 初始化 residual actor
        print("- Residual Actor:")
        for i, layer in enumerate(self.residual_actor):
            is_final = (i == len(self.residual_actor) - 1)
            init_layer(layer, is_final)
        
        # 初始化 residual critic
        print("- Residual Critic:")
        for i, layer in enumerate(self.residual_critic):
            is_final = (i == len(self.residual_critic) - 1)
            init_layer(layer, is_final)
        
        # 验证初始化效果
        self._verify_initialization()
    
    def _verify_initialization(self):
        """验证初始化效果"""
        with torch.no_grad():
            # 测试数据
            dummy_obs = torch.randn(32, self.residual_actor_obs_dim, device=next(self.residual_actor.parameters()).device)
            dummy_obs_critic = torch.randn(32, self.residual_critic_obs_dim, device=next(self.residual_critic.parameters()).device)
            
            # 计算初始输出
            residual_actor_out = self.residual_actor(dummy_obs)
            residual_critic_out = self.residual_critic(dummy_obs_critic)
            
            print("初始化验证:")
            print(f"  Residual Actor 输出范围: [{residual_actor_out.min():.6f}, {residual_actor_out.max():.6f}]")
            print(f"  Residual Actor 平均绝对值: {residual_actor_out.abs().mean():.6f}")
            print(f"  Residual Critic 输出范围: [{residual_critic_out.min():.6f}, {residual_critic_out.max():.6f}]")
            print(f"  Residual Critic 平均绝对值: {residual_critic_out.abs().mean():.6f}")
            
            # 检查是否接近零
            if residual_actor_out.abs().mean() < 1e-5:
                print("  ✅ Residual Actor 成功初始化为接近零")
            else:
                print("  ⚠️  Residual Actor 初始化可能有问题")
                
            if residual_critic_out.abs().mean() < 1e-5:
                print("  ✅ Residual Critic 成功初始化为接近零")
            else:
                print("  ⚠️  Residual Critic 初始化可能有问题")

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
        joint_pos = flattened_obs[:, :, 18:18+29]  # (-1, history_length, 29)
        joint_vel = flattened_obs[:, :, 18+29:18+29+29]  # (-1, history_length, 29)
        other_features = flattened_obs[:, :, 18+29+29:-15]  # (-1, history_length, other_features_dim)
        previliged_features = flattened_obs[:, :, -15:]  # (-1, history_length, 13)
        
        self.total_steps += 1

        residual_action = self.residual_actor(observations)  # (-1, 29)

        original_commands = torch.cat([commands, pose_commands], dim=2)  # (-1, history_length, 18)
        actor_observations = torch.cat([original_commands, joint_pos, joint_vel, other_features], dim=2).reshape(observations.shape[0], -1)

        return actor_observations, residual_action
    
    # 第一种方式：老的critics还是给29维的action打分，只不过新增了一个根据previleged信息生成更真实predicted_command的网络，不太合理因为不知道新增的预测command的网络该怎样用
    def critic_forward(self, observations, inference=False):
        # observations: (-1, history_length*single_frame_observations)
        flattened_obs = observations.reshape(observations.shape[0], self.history_length, -1)

        commands = flattened_obs[:, :, 0:4]
        pose_commands = flattened_obs[:, :, 4:18]
        joint_pos = flattened_obs[:, :, 18:18+29]  # (-1, history_length, 29)
        joint_vel = flattened_obs[:, :, 18+29:18+29+29]  # (-1, history_length, 29)
        other_features = flattened_obs[:, :, 18+29+29:-(15+5)]  # (-1, history_length, other_features_dim)
        previliged_features = flattened_obs[:, :, -(15+5):-5]  # (-1, history_length, 13)
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

        # to verify

        # 现在可以安全加载了
        super().load_state_dict(state_dict, strict=False)
        assert torch.equal(state_dict['actor.0.bias'], self.actor.state_dict()['0.bias'])
        # self._verify_initialization()
        return True
