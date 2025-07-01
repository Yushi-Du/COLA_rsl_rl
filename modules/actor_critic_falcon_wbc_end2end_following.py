import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.utils import resolve_nn_activation

import sys
sys.path.append("/home/yushidu/Documents/Humanoid/isaacgym/python/examples/FALCON")
from humanoidverse.agents.modules.modules import BaseModule
from omegaconf import OmegaConf
from ipdb import set_trace

class ActorCriticFalconWbcEnd2endFollowing(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        lower_body_action_dim=15,   # 需要补充
        upper_body_action_dim=14,   # 需要补充
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        history_length: int = 5,
        num_envs: int = 2048,
        device="cuda:0",
        env=None,
        **kwargs,
    ):
        super().__init__()
        self.device = device

        # 你需要在外部传入 lower_body_action_dim 和 upper_body_action_dim
        assert lower_body_action_dim is not None and upper_body_action_dim is not None, \
            "请在初始化时补充 lower_body_action_dim 和 upper_body_action_dim"

        num_actor_obs = int(num_actor_obs / history_length)

        obs_dim_dict = OmegaConf.create({'actor_obs': num_actor_obs, 'critic_obs': num_critic_obs})

        # 构造上下半身 actor/critic 的 module_config_dict
        module_config_dict_actor_lower = OmegaConf.create({
            "input_dim": ["actor_obs"],
            "history_length": {"actor_obs": history_length},
            "output_dim": [lower_body_action_dim],
            "layer_config": {
                "type": "MLP",
                "hidden_dims": actor_hidden_dims,
                "activation": 'ELU',
            }
        })
        module_config_dict_actor_upper = OmegaConf.create({
            "input_dim": ["actor_obs"],
            "history_length": {"actor_obs": history_length},
            "output_dim": [upper_body_action_dim],
            "layer_config": {
                "type": "MLP",
                "hidden_dims": actor_hidden_dims,
                "activation": 'ELU',
            }
        })
        module_config_dict_critic_lower = OmegaConf.create({
            "input_dim": ["critic_obs"],
            "history_length": {"critic_obs": 1},
            "output_dim": [1],
            "layer_config": {
                "type": "MLP",
                "hidden_dims": critic_hidden_dims,
                "activation": 'ELU',
            }
        })
        module_config_dict_critic_upper = OmegaConf.create({
            "input_dim": ["critic_obs"],
            "history_length": {"critic_obs": 1},
            "output_dim": [1],
            "layer_config": {
                "type": "MLP",
                "hidden_dims": critic_hidden_dims,
                "activation": 'ELU',
            }
        })

        # 创建两个actor和两个critic
        self.actor_module = nn.ModuleDict({
            "lower_body": BaseModule(obs_dim_dict, module_config_dict_actor_lower),
            "upper_body": BaseModule(obs_dim_dict, module_config_dict_actor_upper),
        })
        self.critic_module = nn.ModuleDict({
            "lower_body": BaseModule(obs_dim_dict, module_config_dict_critic_lower),
            "upper_body": BaseModule(obs_dim_dict, module_config_dict_critic_upper),
        })

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.ParameterDict({
                "lower_body": nn.Parameter(init_noise_std * torch.ones(lower_body_action_dim)),
                "upper_body": nn.Parameter(init_noise_std * torch.ones(upper_body_action_dim)),
            })
        elif self.noise_std_type == "log":
            self.log_std = nn.ParameterDict({
                "lower_body": nn.Parameter(torch.log(init_noise_std * torch.ones(lower_body_action_dim))),
                "upper_body": nn.Parameter(torch.log(init_noise_std * torch.ones(upper_body_action_dim))),
            })
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = {}
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        # 返回拼接后的动作均值
        return torch.cat([self.distribution[k].mean for k in ["lower_body", "upper_body"]], dim=-1)

    @property
    def action_std(self):
        return torch.cat([self.distribution[k].stddev for k in ["lower_body", "upper_body"]], dim=-1)

    @property
    def entropy(self):
        return sum(self.distribution[k].entropy().sum(dim=-1) for k in ["lower_body", "upper_body"])

    def update_distribution(self, actor_obs):
        # 分别更新上下半身分布
        for k in ["lower_body", "upper_body"]:
            mean = self.actor_module[k](actor_obs)
            if self.noise_std_type == "scalar":
                std = self.std[k].expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std[k]).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
            self.distribution[k] = Normal(mean, std)

    def act(self, actor_obs, **kwargs):
        self.update_distribution(actor_obs)
        # 拼接上下半身动作
        return torch.cat([self.distribution[k].sample() for k in ["lower_body", "upper_body"]], dim=-1)

    def get_actions_log_prob(self, actions):
        # 拆分actions为上下半身
        lower_dim = self.std["lower_body"].shape[0]
        lower_actions = actions[..., :lower_dim]
        upper_actions = actions[..., lower_dim:]
        log_prob = self.distribution["lower_body"].log_prob(lower_actions).sum(dim=-1) + \
                   self.distribution["upper_body"].log_prob(upper_actions).sum(dim=-1)
        return log_prob

    def act_inference(self, actor_obs):
        # 拼接上下半身动作均值
        return torch.cat([self.actor_module[k](actor_obs) for k in ["lower_body", "upper_body"]], dim=-1)

    def evaluate(self, critic_obs, **kwargs):
        # 返回上下半身critic输出（可拼接或分别返回，视需求而定）
        return torch.cat([self.critic_module[k](critic_obs) for k in ["lower_body", "upper_body"]], dim=-1).mean(dim=-1, keepdim=True)

    def _strip_prefix_from_state_dict(self, state_dict, prefix):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_state_dict[k[len(prefix):]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    def load_state_dict(self, state_dict, strict=True):
        if "actor_model_state_dict" in state_dict and "critic_model_state_dict" in state_dict:
            actor_sd = state_dict["actor_model_state_dict"]
            critic_sd = state_dict["critic_model_state_dict"]
            for k in ["lower_body", "upper_body"]:
                # 只保留 'module.' 开头的参数
                actor_module_sd = self._strip_prefix_from_state_dict(actor_sd[k], "actor_module.")
                self.actor_module[k].load_state_dict(actor_module_sd, strict=False)
                critic_module_sd = self._strip_prefix_from_state_dict(critic_sd[k], "critic_module.")
                self.critic_module[k].load_state_dict(critic_module_sd, strict=False)
            return True
        else:
            return super().load_state_dict(state_dict, strict=strict)