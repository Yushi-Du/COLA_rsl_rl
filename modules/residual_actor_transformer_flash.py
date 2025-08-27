import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
import math

from rsl_rl.utils import resolve_nn_activation


class CustomTransformerEncoderLayer(nn.Module):
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu'):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # Multi-head attention 的线性层
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation
        self.activation = getattr(F, activation) if isinstance(activation, str) else activation
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Multi-head attention
        src2 = self.multi_head_attention(src, src, src, src_mask, src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src
    
    def multi_head_attention(self, query, key, value, attn_mask=None, key_padding_mask=None):
        batch_size, seq_len, _ = query.shape
        
        # Linear projections
        Q = self.q_linear(query)  # (batch_size, seq_len, d_model)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)  # (batch_size, nhead, seq_len, head_dim)
        K = K.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False  # 对于encoder，通常不需要causal mask
        )
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Final linear projection
        output = self.out_linear(attn_output)
        
        return output


class CustomTransformerEncoder(nn.Module):
    
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                encoder_layer.d_model,
                encoder_layer.nhead,
                encoder_layer.linear1.in_features,  # dim_feedforward
                encoder_layer.dropout.p,
                'gelu'
            ) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
    
    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, mask, src_key_padding_mask)
        return output


class TransformerResidualNetworkFlash(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=256, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        dummy_layer = type('DummyLayer', (), {
            'd_model': d_model,
            'nhead': nhead,
            'linear1': type('Linear', (), {'in_features': 4*d_model})(),
            'dropout': type('Dropout', (), {'p': dropout})()
        })()
        
        self.transformer = CustomTransformerEncoder(dummy_layer, num_layers)
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, output_dim)
        
        # 归一化层
        self.layer_norm = nn.LayerNorm(d_model)

    def _init_residual_weights(self):
        
        nn.init.normal_(self.input_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.input_projection.bias)
        
        for layer in self.transformer.layers:
            # Multi-head attention 线性层
            for linear in [layer.q_linear, layer.k_linear, layer.v_linear, layer.out_linear]:
                nn.init.xavier_uniform_(linear.weight, gain=0.1)
                nn.init.zeros_(linear.bias)
            
            # Feed forward
            nn.init.xavier_uniform_(layer.linear1.weight, gain=0.1)
            nn.init.zeros_(layer.linear1.bias)
            nn.init.xavier_uniform_(layer.linear2.weight, gain=0.1)
            nn.init.zeros_(layer.linear2.bias)
            
            # Layer norms
            nn.init.ones_(layer.norm1.weight)
            nn.init.zeros_(layer.norm1.bias)
            nn.init.ones_(layer.norm2.weight)
            nn.init.zeros_(layer.norm2.bias)
        
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        
        print("Transformer Residual 网络初始化完成:")
        print(f"  - 输入投影: 小随机初始化 (std=0.02)")
        print(f"  - Transformer 层: 缩放的 Xavier 初始化 (gain=0.1)")
        print(f"  - 输出投影: 零初始化")
        print(f"  - 使用 scaled_dot_product_attention 优化")
        
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