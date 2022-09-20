import torch
from torch import nn
from torch.nn import functional


class NonLocalSelfAttention(nn.Module):
    def __init__(self, channel, h, w):
        super().__init__()
        self.h, self.w = h, w
        self.hidden_channel = channel//2
        self.key_kernel = nn.Conv2d(channel, self.hidden_channel, 1, 1)
        self.query_kernel = nn.Conv2d(channel, self.hidden_channel, 1, 1)
        self.value_kernel = nn.Conv2d(channel, self.hidden_channel, 1, 1)
        self.mask_kernel = nn.Conv2d(self.hidden_channel, channel, 1, 1)

    def forward(self, feature):
        n = feature.shape[0]

        k, q, v = self.key_kernel(feature), self.query_kernel(feature), self.value_kernel(feature)
        k, q, v = k.transpose(1, 2).transpose(2, 3), q.transpose(1, 2).transpose(2, 3), v.transpose(1, 2).transpose(2, 3)
        k = k.reshape(n * self.h * self.w, self.hidden_channel)
        q = q.reshape(n * self.h * self.w, self.hidden_channel).T
        v = v.reshape(n * self.h * self.w, self.hidden_channel)
        mask = torch.matmul(functional.softmax(torch.matmul(k, q)), v).reshape(n, self.h, self.w, self.hidden_channel)
        mask = mask.transpose(2, 3).transpose(1, 2).transpose(2, 3)
        mask = self.mask_kernel(mask)

        return mask + feature


class MergeBatchSelfAttention(nn.Module):
    def __init__(self, channel, out_channel, len_in, len_out, keep_dim=False):
        super().__init__()
        self.keep_dim = keep_dim

        self.channel = channel
        self.out_channel = out_channel
        self.len_in = len_in
        self.len_out = len_out
        self.hidden_channel = channel//2

        self.token_mlp = nn.Linear(len_in, len_out)
        self.feature_mlp = nn.Linear(channel, out_channel)
        self.key_mlp = nn.Linear(channel, self.hidden_channel)
        self.query_mlp = nn.Linear(channel, self.hidden_channel)
        self.value_mlp = nn.Linear(channel, self.hidden_channel)

        self.attention = nn.MultiheadAttention(self.hidden_channel, 4)
        self.out_mlp = nn.Linear(self.hidden_channel, out_channel)
        self.norm_layer = nn.BatchNorm1d(num_features=out_channel, affine=False)

    def forward(self, feature):  # N X L X C
        # print(feature.shape)
        n, l, c = feature.shape
        feature = self.token_mlp(feature.transpose(1, 2)).transpose(1, 2)   # N X len_out X C
        feature = feature.reshape(n * self.len_out, c)
        # print(f"Feature: {feature.shape}")

        k, q, v = self.key_mlp(feature), self.query_mlp(feature), self.value_mlp(feature)
        out, _ = self.attention(q, k, v, need_weights=False)   # N * len_out X C
        # print(f"Out: {out.shape}")
        out = self.out_mlp(out)
        # print(f"Out: {out.shape}")
        feature = self.feature_mlp(feature)
        # print(f"Feature: {feature.shape}")
        out = out + feature   # N * len_out X C_out
        out = self.norm_layer(out) if out.shape[0] != 1 else out
        if self.keep_dim:
            out = out.reshape(-1, self.len_out, self.out_channel)
        return out


class MHSelfAttention(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.hidden_channel = channel//2
        self.key_mlp = nn.Linear(channel, self.hidden_channel)
        self.query_mlp = nn.Linear(channel, self.hidden_channel)
        self.value_mlp = nn.Linear(channel, self.hidden_channel)

        self.attention = nn.MultiheadAttention(self.hidden_channel, 4, batch_first=True)
        self.out_mlp = nn.Linear(self.hidden_channel, channel)
        self.norm_layer = nn.BatchNorm1d(num_features=channel, affine=False)

    def forward(self, feature):  # N X L X C
        # print(f"Feature: {feature.shape}")
        k, q, v = self.key_mlp(feature), self.query_mlp(feature), self.value_mlp(feature)
        # N X L X C, N X L X C, N X L X C
        # print(f"Key: {k.shape}")
        # print(f"Query: {q.shape}")
        # print(f"Value: {v.shape}")
        # print(f"K-Q-V: {out.shape}")
        out, _ = self.attention(q, k, v, need_weights=False)
        out = self.out_mlp(out)
        # print(f"Out: {out.shape}")
        out = self.norm_layer(out.transpose(1, 2)).transpose(1, 2) + feature  # N X L X C
        # print(f"Out: {out.shape}")
        # print()
        return out


class SelfAttentionStage(nn.Module):
    def __init__(self, channel, n):
        super().__init__()
        self.n = n
        self.blocks = nn.Sequential(*[MHSelfAttention(channel) for _ in range(n)])

    def forward(self, feature):
        feature = self.blocks(feature)
        return feature
