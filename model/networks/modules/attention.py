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
    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        self.hidden_channel = channel//2
        self.key_mlp = nn.Linear(channel, self.hidden_channel)
        self.query_mlp = nn.Linear(channel, self.hidden_channel)
        self.value_mlp = nn.Linear(channel, self.hidden_channel)
        self.out_mlp = nn.Linear(self.hidden_channel, channel)
        self.norm_layer = nn.BatchNorm1d(num_features=channel, affine=False)

    def forward(self, feature):
        k, q, v = self.key_mlp(feature).T, self.query_mlp(feature), torch.sum(self.value_mlp(feature), dim=0)
        kq = torch.matmul(k, q)
        out = torch.matmul(functional.softmax(kq), v).reshape(1, kq.shape[0])
        out = self.out_mlp(out) + torch.sum(feature, dim=0).reshape(1, self.channel)
        out = self.norm_layer(out) if out.shape[0] != 1 else out
        return out


class SelfAttention(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.hidden_channel = channel//2
        self.key_mlp = nn.Linear(channel, self.hidden_channel)
        self.query_mlp = nn.Linear(channel, self.hidden_channel)
        self.value_mlp = nn.Linear(channel, self.hidden_channel)
        self.out_mlp = nn.Linear(self.hidden_channel, channel)
        self.norm_layer = nn.BatchNorm1d(num_features=channel, affine=False)

    def forward(self, feature):
        # print(f"Feature: {feature.shape}")
        k, q, v = self.key_mlp(feature).T, self.query_mlp(feature), self.value_mlp(feature).T
        # print(f"Key: {k.shape}")
        # print(f"Query: {q.shape}")
        # print(f"Value: {v.shape}")
        kq = torch.matmul(k, q)
        # print(f"Key-Query: {q.shape}")
        out = torch.matmul(functional.softmax(kq), v).reshape(v.shape[1], kq.shape[0])
        # print(f"K-Q-V: {out.shape}")
        out = self.out_mlp(out)+feature
        out = self.norm_layer(out) if out.shape[0] != 1 else out
        # print(f"Out: {out.shape}")
        # print()
        return out


class SelfAttentionStage(nn.Module):
    def __init__(self, channel, n):
        super().__init__()
        self.n = n
        self.blocks = nn.Sequential(*[SelfAttention(channel) for _ in range(n)])

    def forward(self, feature):
        feature = self.blocks(feature)
        return feature
