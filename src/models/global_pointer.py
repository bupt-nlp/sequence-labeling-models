from __future__ import annotations

import torch
from torch import nn
from torch._C import dtype

class SinusoidalPositionEmbedding(nn.Module):
    """定义Sin-Cos位置Embedding
    """
    def __init__(
        self, output_dim: int, merge_mode: str ='add', custom_position_ids: int=False, **kwargs
    ):
        super(SinusoidalPositionEmbedding, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

        self.float_type = torch.float64

    def forward(self, inputs: torch.Tensor):
        """如果custom_position_ids，那么第二个输入为自定义的位置id
        """
        if self.custom_position_ids:
            seq_len = inputs.shape[1]
            inputs, position_ids = inputs

            # define the position ids
            position_ids: torch.Tensor = position_ids
            if 'float' not in str(position_ids.dtype):
                position_ids = position_ids.float()
        else:
            input_shape = inputs.shape
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = torch.arange(0, seq_len, dtype=self.float_type)[None]

        indices = torch.arange(0, self.output_dim // 2, dtype=self.float_type)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)

        embeddings: torch.Tensor = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))

        if self.merge_mode == 'add':
            return inputs + embeddings
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0)
        elif self.merge_mode == 'zero':
            return embeddings
        else:
            if not self.custom_position_ids:
                embeddings
                embeddings = torch.tile(embeddings, [batch_size, 1, 1])
            return torch.cat([inputs, embeddings])

    def compute_output_shape(self, input_shape):
        if self.custom_position_ids:
            input_shape = input_shape[0]

        if self.merge_mode in ['add', 'mul', 'zero']:
            return input_shape[:2] + (self.output_dim,)
        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim,)


class GlobalPointer(nn.Module):
    def __init__(self, head_num: int, input_dim: int, head_size: int):
        self.dense = nn.Linear(in_features=input_dim, out_features=head_size)
        self.head_size: int = head_size
        self.head_num: int = head_num
    
    def forward(self, inputs: torch.Tensor, mask: torch.Tensor = None): 
        # 输入变换
        inputs = self.dense(inputs)
        inputs = torch.split(inputs, self.head_num, dim=-1)
        inputs = torch.stack(inputs, dim=-2)
        qw, kw = inputs[..., :self.head_size], inputs[..., self.head_size:]

        # RoPE编码
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.head_size, 'zero')(inputs)
            cos_pos = K.repeat_elements(pos[..., None, 1::2], 2, -1)
            sin_pos = K.repeat_elements(pos[..., None, ::2], 2, -1)
            qw2 = K.stack([-qw[..., 1::2], qw[..., ::2]], 4)
            qw2 = K.reshape(qw2, K.shape(qw))
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = K.stack([-kw[..., 1::2], kw[..., ::2]], 4)
            kw2 = K.reshape(kw2, K.shape(kw))
            kw = kw * cos_pos + kw2 * sin_pos
        # 计算内积
        logits = tf.einsum('bmhd,bnhd->bhmn', qw, kw)
        # 排除padding
        logits = sequence_masking(logits, mask, '-inf', 2)
        logits = sequence_masking(logits, mask, '-inf', 3)
        # 排除下三角
        mask = tf.matrix_band_part(K.ones_like(logits), 0, -1)
        logits = logits - (1 - mask) * 1e12
        # scale返回
        return logits / self.head_size**0.5

        