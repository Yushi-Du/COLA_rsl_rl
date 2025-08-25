import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
import math  # 新增

from rsl_rl.utils import resolve_nn_activation


class TransformerResidualNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=256, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, output_dim)
        
        # 归一化层
        self.layer_norm = nn.LayerNorm(d_model)

    def _init_residual_weights(self):
        """专门为 residual 网络设计的初始化"""
        
        # 1. 输入投影：小随机初始化
        nn.init.normal_(self.input_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.input_projection.bias)
        
        # 2. Transformer 层：标准初始化但缩放
        for layer in self.transformer.layers:
            # Multi-head attention
            for param in layer.self_attn.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param, gain=0.1)  # 小的 gain
                else:
                    nn.init.zeros_(param)
            
            # Feed forward
            nn.init.xavier_uniform_(layer.linear1.weight, gain=0.1)
            nn.init.zeros_(layer.linear1.bias)
            nn.init.xavier_uniform_(layer.linear2.weight, gain=0.1)
            nn.init.zeros_(layer.linear2.bias)
            
            # Layer norms: 标准初始化
            nn.init.ones_(layer.norm1.weight)
            nn.init.zeros_(layer.norm1.bias)
            nn.init.ones_(layer.norm2.weight)
            nn.init.zeros_(layer.norm2.bias)
        
        # 3. 🔥 只有输出层完全零初始化
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        
        print("Transformer Residual 网络初始化完成:")
        print(f"  - 输入投影: 小随机初始化 (std=0.02)")
        print(f"  - Transformer 层: 缩放的 Xavier 初始化 (gain=0.1)")
        print(f"  - 输出投影: 零初始化")
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim) 或 (batch_size, input_dim)
        
        if x.dim() == 2:
            # 如果输入是 2D，添加序列维度
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # 输入投影
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # Transformer 编码
        x = self.transformer(x)  # (batch_size, seq_len, d_model)
        
        # 归一化
        x = self.layer_norm(x)
        
        # 取最后一个时间步或平均池化
        if x.size(1) > 1:
            x = x.mean(dim=1)  # 平均池化: (batch_size, d_model)
        else:
            x = x.squeeze(1)   # (batch_size, d_model)
        
        # 输出投影
        output = self.output_projection(x)  # (batch_size, output_dim)
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)