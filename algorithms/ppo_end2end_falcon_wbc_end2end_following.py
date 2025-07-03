# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain

from rsl_rl.modules import ActorCriticFalconWbcEnd2endFollowing
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import string_to_callable

from ipdb import set_trace


class PPO_FalconWbcEnd2endFollowing:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: ActorCriticFalconWbcEnd2endFollowing
    """The actor critic module."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg is not None:
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_cfg.get("learning_rate", 1e-3))
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        # 6_3: 看到这里的policy就是ActorCritic, 所以只要是ActorCritic中有的参数就会被更新
        # Create optimizer
        # self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.optimizer_lower = optim.Adam(
            list(self.policy.actor_module["lower_body"].parameters()) +
            list(self.policy.critic_module["lower_body"].parameters()),
            lr=learning_rate
        )
        self.optimizer_upper = optim.Adam(
            list(self.policy.actor_module["upper_body"].parameters()) +
            list(self.policy.critic_module["upper_body"].parameters()),
            lr=learning_rate
        )
        # Create rollout storage
        self.rollout_storage_lower: RolloutStorage = None
        self.rollout_storage_upper: RolloutStorage = None
        self.transition_lower = RolloutStorage.Transition()
        self.transition_upper = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_lower_shape, actions_upper_shape
    ):
        # create memory for RND as well :)
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        # create rollout storage
        self.rollout_storage_lower = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_lower_shape,
            rnd_state_shape,
            self.device,
        )
        self.rollout_storage_upper = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_upper_shape,
            rnd_state_shape,
            self.device,
        )

    def act(self, obs, critic_obs, inference=False):
        if self.policy.is_recurrent:
            raise NotImplementedError(
                "Recurrent policies are not supported in PPO_FalconWbcEnd2endFollowing. "
                "Please use PPO_FalconWbcEnd2endFollowingRecurrent instead."
            )
            pass
        # compute the actions and values
        self.transition_lower.actions, self.transition_upper.actions = map(lambda x: x.detach(), self.policy.act(obs))
        self.transition_lower.values, self.transition_upper.values = map(lambda x: x.detach(), self.policy.evaluate(critic_obs))
        self.transition_lower.actions_log_prob, self.transition_upper.actions_log_prob = map(lambda x: x.detach(), self.policy.get_actions_log_prob(
            self.transition_lower.actions, self.transition_upper.actions
        ))
        self.transition_lower.action_mean, self.transition_upper.action_mean = map(lambda x: x.detach(), self.policy.action_mean)
        self.transition_lower.action_sigma, self.transition_upper.action_sigma = map(lambda x: x.detach(), self.policy.action_std)
        # need to record obs and critic_obs before env.step()
        self.transition_lower.observations = obs
        self.transition_upper.observations = obs
        self.transition_lower.privileged_observations = critic_obs
        self.transition_upper.privileged_observations = critic_obs
        return self.transition_lower.actions, self.transition_upper.actions

    def process_env_step(self, rewards_upper_body, rewards_lower_body, dones, infos):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition_lower.rewards = rewards_lower_body.clone()
        self.transition_upper.rewards = rewards_upper_body.clone()
        self.transition_lower.dones = dones
        self.transition_upper.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            raise NotImplementedError(
                "RND is not implemented for PPO_FalconWbcEnd2endFollowing. "
                "Please implement it if you want to use RND with PPO_FalconWbcEnd2endFollowing."
            )
            pass

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition_lower.rewards += self.gamma * torch.squeeze(
                (self.transition_lower.values) * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
            self.transition_upper.rewards += self.gamma * torch.squeeze(
                (self.transition_upper.values) * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # record the transition
        self.rollout_storage_lower.add_transitions(self.transition_lower)
        self.rollout_storage_upper.add_transitions(self.transition_upper)
        self.transition_lower.clear()
        self.transition_upper.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs, inference=False):
        # compute value for the last step
        last_values_lower, last_values_upper = map(lambda x: x.detach(), self.policy.evaluate(last_critic_obs))
        self.rollout_storage_lower.compute_returns(
            last_values_lower, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )
        self.rollout_storage_upper.compute_returns(
            last_values_upper, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):  # noqa: C901
        mean_value_loss = {"lower_body": 0, "upper_body": 0}
        mean_surrogate_loss = {"lower_body": 0, "upper_body": 0}
        mean_entropy = {"lower_body": 0, "upper_body": 0}
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        if self.symmetry:
            mean_symmetry_loss = {"lower_body": 0, "upper_body": 0}
        else:
            mean_symmetry_loss = None

        if self.policy.is_recurrent:
            raise NotImplementedError(
                "Recurrent policies are not supported in PPO_FalconWbcEnd2endFollowing. "
                "Please use PPO_FalconWbcEnd2endFollowingRecurrent instead."
            )
        else:
            generator_lower = self.rollout_storage_lower.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            generator_upper = self.rollout_storage_upper.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (batch_lower, batch_upper) in zip(generator_lower, generator_upper):
            # unpack lower
            (
                obs_batch_lower,
                critic_obs_batch_lower,
                actions_batch_lower,
                target_values_batch_lower,
                advantages_batch_lower,
                returns_batch_lower,
                old_actions_log_prob_batch_lower,
                old_mu_batch_lower,
                old_sigma_batch_lower,
                hid_states_batch_lower,
                masks_batch_lower,
                rnd_state_batch_lower,
            ) = batch_lower
            # unpack upper
            (
                obs_batch_upper,
                critic_obs_batch_upper,
                actions_batch_upper,
                target_values_batch_upper,
                advantages_batch_upper,
                returns_batch_upper,
                old_actions_log_prob_batch_upper,
                old_mu_batch_upper,
                old_sigma_batch_upper,
                hid_states_batch_upper,
                masks_batch_upper,
                rnd_state_batch_upper,
            ) = batch_upper

            # 归一化 advantage
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch_lower = (advantages_batch_lower - advantages_batch_lower.mean()) / (advantages_batch_lower.std() + 1e-8)
                    advantages_batch_upper = (advantages_batch_upper - advantages_batch_upper.mean()) / (advantages_batch_upper.std() + 1e-8)

            # 数据增强
            num_aug = 1
            original_batch_size = obs_batch_lower.shape[0]
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch_lower, actions_batch_lower = data_augmentation_func(
                    obs=obs_batch_lower, actions=actions_batch_lower, env=self.symmetry["_env"], obs_type="policy"
                )
                obs_batch_upper, actions_batch_upper = data_augmentation_func(
                    obs=obs_batch_upper, actions=actions_batch_upper, env=self.symmetry["_env"], obs_type="policy"
                )
                critic_obs_batch_lower, _ = data_augmentation_func(
                    obs=critic_obs_batch_lower, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                critic_obs_batch_upper, _ = data_augmentation_func(
                    obs=critic_obs_batch_upper, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                num_aug = int(obs_batch_lower.shape[0] / original_batch_size)
                old_actions_log_prob_batch_lower = old_actions_log_prob_batch_lower.repeat(num_aug, 1)
                old_actions_log_prob_batch_upper = old_actions_log_prob_batch_upper.repeat(num_aug, 1)
                target_values_batch_lower = target_values_batch_lower.repeat(num_aug, 1)
                target_values_batch_upper = target_values_batch_upper.repeat(num_aug, 1)
                advantages_batch_lower = advantages_batch_lower.repeat(num_aug, 1)
                advantages_batch_upper = advantages_batch_upper.repeat(num_aug, 1)
                returns_batch_lower = returns_batch_lower.repeat(num_aug, 1)
                returns_batch_upper = returns_batch_upper.repeat(num_aug, 1)

            # 一次forward，获得所有分支结果
            # 注意：这里假设obs/critic_obs分别输入lower/upper的batch
            # get_actions_log_prob、evaluate等方法都返回(lower, upper)
            self.policy.act(obs_batch_lower)
            log_prob_lower, log_prob_upper = self.policy.get_actions_log_prob(actions_batch_lower, actions_batch_upper)
            value_lower, value_upper = self.policy.evaluate(critic_obs_batch_lower)
            mean_lower, mean_upper = self.policy.action_mean
            std_lower, std_upper = self.policy.action_std
            entropy_lower = self.policy.distribution["lower_body"].entropy().mean()
            entropy_upper = self.policy.distribution["upper_body"].entropy().mean()

            # 分别计算lower/upper的loss
            for branch, (
                log_prob, value, mean, std, entropy, actions_batch, target_values_batch, advantages_batch, returns_batch,
                old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, rnd_state_batch
            ) in {
                "lower_body": (
                    log_prob_lower, value_lower, mean_lower, std_lower, entropy_lower,
                    actions_batch_lower, target_values_batch_lower, advantages_batch_lower, returns_batch_lower,
                    old_actions_log_prob_batch_lower, old_mu_batch_lower, old_sigma_batch_lower, rnd_state_batch_lower
                ),
                "upper_body": (
                    log_prob_upper, value_upper, mean_upper, std_upper, entropy_upper,
                    actions_batch_upper, target_values_batch_upper, advantages_batch_upper, returns_batch_upper,
                    old_actions_log_prob_batch_upper, old_mu_batch_upper, old_sigma_batch_upper, rnd_state_batch_upper
                ),
            }.items():
                # KL
                if self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(std / old_sigma_batch + 1.0e-5)
                            + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mean))
                            / (2.0 * torch.square(std))
                            - 0.5,
                            axis=-1,
                        )
                        kl_mean = torch.mean(kl)
                        if self.is_multi_gpu:
                            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                            kl_mean /= self.gpu_world_size
                        if self.gpu_global_rank == 0:
                            if kl_mean > self.desired_kl * 2.0:
                                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                                self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        if self.is_multi_gpu:
                            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                            torch.distributed.broadcast(lr_tensor, src=0)
                            self.learning_rate = lr_tensor.item()
                        for param_group in self.optimizer_lower.param_groups:
                            param_group["lr"] = self.learning_rate
                        for param_group in self.optimizer_upper.param_groups:
                            param_group["lr"] = self.learning_rate

                # Surrogate loss
                ratio = torch.exp(log_prob - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value - target_values_batch).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (value - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                # Symmetry loss
                if self.symmetry:
                    if not self.symmetry["use_data_augmentation"]:
                        data_augmentation_func = self.symmetry["data_augmentation_func"]
                        obs_aug, _ = data_augmentation_func(
                            obs=obs_batch_lower if branch == "lower_body" else obs_batch_upper,
                            actions=None, env=self.symmetry["_env"], obs_type="policy"
                        )
                        num_aug = int(obs_aug.shape[0] / original_batch_size)
                    mean_actions_lower, mean_actions_upper = self.policy.act_inference(
                        obs_batch_lower.detach().clone(), obs_batch_upper.detach().clone()
                    )
                    action_mean_orig = mean_actions_lower if branch == "lower_body" else mean_actions_upper
                    _, actions_mean_symm_batch = data_augmentation_func(
                        obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                    )
                    mse_loss = torch.nn.MSELoss()
                    symmetry_loss = mse_loss(
                        action_mean_orig[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                    )
                    if self.symmetry["use_mirror_loss"]:
                        loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                    else:
                        symmetry_loss = symmetry_loss.detach()
                    mean_symmetry_loss[branch] += symmetry_loss.item()

                # Random Network Distillation loss
                if self.rnd:
                    predicted_embedding = self.rnd.predictor(rnd_state_batch)
                    target_embedding = self.rnd.target(rnd_state_batch).detach()
                    mseloss = torch.nn.MSELoss()
                    rnd_loss = mseloss(predicted_embedding, target_embedding)

                # Compute the gradients
                # 优化器选择
                if branch == "lower_body":
                    self.optimizer_lower.zero_grad()
                else:
                    self.optimizer_upper.zero_grad()

                loss.backward()
                # if self.rnd:
                #     raise NotImplementedError(
                #         "RND is not implemented for PPO_FalconWbcEnd2endFollowing. "
                #         "Please implement it if you want to use RND with PPO_FalconWbcEnd2endFollowing."
                #     )
                #     self.rnd_optimizer.zero_grad()
                #     rnd_loss.backward()
                # if self.is_multi_gpu:
                #     self.reduce_parameters()
                # 梯度裁剪
                if branch == "lower_body":
                    nn.utils.clip_grad_norm_(self.policy.actor_module["lower_body"].parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.policy.critic_module["lower_body"].parameters(), self.max_grad_norm)
                    self.optimizer_lower.step()
                else:
                    nn.utils.clip_grad_norm_(self.policy.actor_module["upper_body"].parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.policy.critic_module["upper_body"].parameters(), self.max_grad_norm)
                    self.optimizer_upper.step()
                # if self.rnd_optimizer:
                #     self.rnd_optimizer.step()

                mean_value_loss[branch] += value_loss.item()
                mean_surrogate_loss[branch] += surrogate_loss.item()
                mean_entropy[branch] += entropy.item()
                if mean_rnd_loss is not None:
                    mean_rnd_loss += rnd_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        for branch in ["lower_body", "upper_body"]:
            mean_value_loss[branch] /= num_updates
            mean_surrogate_loss[branch] /= num_updates
            mean_entropy[branch] /= num_updates
            if mean_symmetry_loss is not None:
                mean_symmetry_loss[branch] /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= (2 * num_updates)
        self.rollout_storage_lower.clear()
        self.rollout_storage_upper.clear()

        loss_dict = {
            "value_function_lower": mean_value_loss["lower_body"],
            "surrogate_lower": mean_surrogate_loss["lower_body"],
            "entropy_lower": mean_entropy["lower_body"],
            "value_function_upper": mean_value_loss["upper_body"],
            "surrogate_upper": mean_surrogate_loss["upper_body"],
            "entropy_upper": mean_entropy["upper_body"],
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry_lower"] = mean_symmetry_loss["lower_body"]
            loss_dict["symmetry_upper"] = mean_symmetry_loss["upper_body"]

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
