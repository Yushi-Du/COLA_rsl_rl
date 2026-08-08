import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic import ActorCritic
from .sensor_transformer import SensorTransformer
from ipdb import set_trace

class ActorCriticTransformer(ActorCritic):
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, **kwargs):
        # 假设tactile部分在obs的最后N维
        self.transformer_seq_len = 6
        self.num_sensors = 48
        self.transformer_output_dim = 3
        self.history_len = 10
        self.tactile_dim = self.num_sensors * 3  # 你实际的tactile观测维度
        self.mono_actor_obs_dim = int(num_actor_obs/10)
        self.mono_critic_obs_dim = int(num_critic_obs/10)
        self.other_obs_dim_a = int(self.mono_actor_obs_dim - self.tactile_dim * self.transformer_seq_len)

        super().__init__(num_actor_obs-self.history_len*(self.transformer_seq_len*self.tactile_dim-self.transformer_output_dim), 
            num_critic_obs-self.history_len*(self.transformer_seq_len*self.tactile_dim-self.transformer_output_dim), num_actions, 
            **kwargs)

        self.sensor_transformer = SensorTransformer(
            input_dim=3, seq_len=self.transformer_seq_len, num_sensors=self.num_sensors, d_model=128, nhead=8, num_layers=4, dim_feedforward=256, output_dim=self.transformer_output_dim, dropout=0.1
        )

    def actor_forward(self, obs):
        # obs: (batch, history_len * obs_dim)
        batch_size = obs.shape[0]
        tactile_obs = obs.reshape(batch_size, self.history_len, -1)[:, :, -self.tactile_dim*self.transformer_seq_len:]  # (batch, history_len, tactile_dim)
        tactile_obs = tactile_obs.reshape(batch_size, self.history_len, self.transformer_seq_len, self.tactile_dim)
        tactile_obs = tactile_obs.reshape(batch_size, self.history_len, self.transformer_seq_len, self.num_sensors, 3)  # (batch, history_len, transformer_seq_len, num_sensors, 3)
        other_obs = obs.reshape(batch_size, self.history_len, -1)[:, :, 0:self.other_obs_dim_a]  # (batch, history_len, other_obs_dim)

        # 分批送入 Transformer
        tactile_obs_flat = tactile_obs.reshape(batch_size * self.history_len, self.transformer_seq_len, self.num_sensors, 3)
        split_size = 128  # 可根据显存调整
        chunks = torch.split(tactile_obs_flat, split_size, dim=0)
        outputs = []
        for chunk in chunks:
            outputs.append(self.sensor_transformer(chunk))
        tactile_feat = torch.cat(outputs, dim=0)  # (batch_size * history_len, output_dim)
        tactile_feat = tactile_feat.reshape(batch_size, self.history_len, -1)

        obs_cat = torch.cat([other_obs, tactile_feat], dim=2)
        obs_cat = obs_cat.reshape(batch_size, -1)
        return self.actor(obs_cat)

    def act(self, observations, **kwargs):
        self.update_distribution(self.actor_forward(observations))
        return self.distribution.sample()

    def act_inference(self, observations):
        return self.actor_forward(observations)

    def update_distribution(self, actor_out):
        # actor_out: (batch, action_dim)
        mean = actor_out
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = torch.distributions.Normal(mean, std)

    def evaluate(self, critic_observations, **kwargs):
        # critic_observations: (batch, obs_dim)
        # critic_observation每单独一条的数据分布形式：（0, other_obs_dim_a, mono_actor_obs_dim, mono_critic_obs_dim）
        batch_size = critic_observations.shape[0]
        tactile_obs = critic_observations.reshape(batch_size, self.history_len, -1)[:, :, self.other_obs_dim_a:self.mono_actor_obs_dim]  # (batch, history_len, tactile_dim)
        tactile_obs = tactile_obs.reshape(batch_size, self.history_len, self.transformer_seq_len, self.tactile_dim)
        tactile_obs = tactile_obs.reshape(batch_size, self.history_len, self.transformer_seq_len, self.num_sensors, 3)  # (batch, history_len, transformer_seq_len, num_sensors, 3)
        other_obs_a = critic_observations.reshape(batch_size, self.history_len, -1)[:, :, 0:self.other_obs_dim_a]  # (batch, history_len, other_obs_dim)
        other_obs_c = critic_observations.reshape(batch_size, self.history_len, -1)[:, :, self.mono_actor_obs_dim:]
        
        # 分批送入 Transformer
        tactile_obs_flat = tactile_obs.reshape(batch_size * self.history_len, self.transformer_seq_len, self.num_sensors, 3)
        split_size = 128  # 可根据显存调整
        chunks = torch.split(tactile_obs_flat, split_size, dim=0)
        outputs = []
        for chunk in chunks:
            outputs.append(self.sensor_transformer(chunk))
        tactile_feat = torch.cat(outputs, dim=0)  # (batch_size * history_len, output_dim)
        tactile_feat = tactile_feat.reshape(batch_size, self.history_len, -1)
        
        obs_cat = torch.cat([other_obs_a, tactile_feat, other_obs_c], dim=2)
        obs_cat = obs_cat.reshape(batch_size, -1)
        return self.critic(obs_cat)
