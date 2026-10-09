import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SequentialCausalDebiasingModule(nn.Module):
    """
    """

    def __init__(self, hidden_dim, item_num, max_len, pop_tensor=None, exposure_tensor=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.item_num = item_num
        self.max_len = max_len
        self.pop_emb = nn.Embedding(item_num + 1, hidden_dim, padding_idx=0)  # 流行度嵌入
        self.exposure_emb = nn.Embedding(item_num + 1, hidden_dim, padding_idx=0)  # 曝光偏差嵌入
        self.time_bias_emb = nn.Embedding(max_len + 1, hidden_dim, padding_idx=0)  # 时间偏差嵌入

        self.direct_path = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.indirect_path = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )


        self.ate_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )


        self.time_decay = nn.Parameter(torch.tensor(0.01))  # 可学习的时间衰减系数

        if pop_tensor is not None:
            self.register_buffer('pop_tensor', pop_tensor)
            with torch.no_grad():
                self.pop_emb.weight.copy_(self._init_emb_from_tensor(self.pop_tensor, self.pop_emb.weight))
        if exposure_tensor is not None:
            self.register_buffer('exposure_tensor', exposure_tensor)
            with torch.no_grad():
                self.exposure_emb.weight.copy_(
                    self._init_emb_from_tensor(self.exposure_tensor, self.exposure_emb.weight)
                )

    def _init_emb_from_tensor(self, tensor, emb_weight):
        if tensor.ndim != 1:
            raise ValueError("initialization features must be a one-dimensional tensor")
        values = tensor.detach().to(device=emb_weight.device, dtype=emb_weight.dtype)
        # 固定单位方向，仅保留标量大小；逐行 L2 归一化会抹去频率差异。
        initialized = torch.zeros_like(emb_weight)
        min_len = min(values.shape[0], emb_weight.shape[0])
        initialized[:min_len] = values[:min_len].unsqueeze(-1).expand(-1, self.hidden_dim) / (self.hidden_dim ** 0.5)
        initialized[0] = 0.0
        return initialized

    def _compute_time_decay_weight(self, seq_len, batch_size, positions):
        pos_expanded = positions.unsqueeze(-1).repeat(1, 1, self.hidden_dim)  # [B, T, D]
        decay_weight = torch.exp(-self.time_decay * (self.max_len - pos_expanded))
        return decay_weight  # [B, T, D]

    def forward(self, x, item_ids, positions=None, item_feat=None):
        """

        """
        seq_len, batch_size, d_model = x.shape

        x_t = x.transpose(0, 1)  # [B, T, D]
        item_ids_t = item_ids  # [B, T]
        valid_positions = item_ids_t.ne(0)
        padding_mask_expanded = (~valid_positions).unsqueeze(-1)

        pop_feat = self.pop_emb(item_ids_t).masked_fill(padding_mask_expanded, 0.0)  # [B, T, D]
        exposure_feat = self.exposure_emb(item_ids_t).masked_fill(padding_mask_expanded, 0.0)  # [B, T, D]

        if item_feat is not None:
            cos_sim = F.cosine_similarity(item_feat, exposure_feat, dim=-1)
            # 仅对真实交互取平均；全 padding 时返回可反传的零损失。
            valid_count = valid_positions.sum().clamp_min(1)
            ortho_loss = (cos_sim ** 2).masked_fill(~valid_positions, 0.0).sum() / valid_count
        else:
            ortho_loss = x.new_zeros(())

        if positions is not None:
            time_decay_weight = self._compute_time_decay_weight(seq_len, batch_size, positions)
            pop_feat = pop_feat * time_decay_weight
            exposure_feat = exposure_feat * time_decay_weight

        direct_bias = self.direct_path(torch.cat([pop_feat, exposure_feat], dim=-1))  # [B, T, D]
        if positions is None:
            positions = torch.arange(1, seq_len + 1, device=x.device).unsqueeze(0).expand(batch_size, -1)
        time_ids = torch.clamp(positions, min=1, max=self.max_len).masked_fill(~valid_positions, 0)
        time_bias = self.time_bias_emb(time_ids).masked_fill(padding_mask_expanded, 0.0)  # [B, T, D]



        indirect_bias = self.indirect_path(torch.cat([pop_feat, exposure_feat, time_bias], dim=-1))  # [B, T, D]

        total_bias = (0.6 * direct_bias + 0.4 * indirect_bias).masked_fill(padding_mask_expanded, 0.0)


        x_counterfactual = x_t - total_bias  # [B, T, D]

        gate_logit = self.ate_estimator(torch.cat([x_t, x_counterfactual], dim=-1))  # [B, T, D]
        debias_gate = torch.sigmoid(gate_logit)  # [B, T, D]

        x_debiased_t = (x_t - debias_gate * total_bias).masked_fill(padding_mask_expanded, 0.0)  # [B, T, D]

        ate = debias_gate.mean(dim=-1).masked_fill(~valid_positions, 0.0)

        x_debiased = x_debiased_t.transpose(0, 1)  # [T, B, D]
        return x_debiased, ate, ortho_loss
