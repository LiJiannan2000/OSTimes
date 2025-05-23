import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x_alpha, x_beta):
        B, N, C = x_alpha.shape
        q = self.q(x_alpha).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = (
            self.kv(x_beta)
            .reshape(B, N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = (
            kv[0],
            kv[1],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossTrans(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.crossAttention = CrossAttention(dim, heads=heads, dropout_rate=attn_dropout_rate)


    def forward(self, x_alpha, x_beta):
        return self.dropout(self.crossAttention(x_alpha, x_beta)) + x_beta


class SelfTrans(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.selfAttention = SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate)


    def forward(self, x):
        return self.dropout(self.selfAttention(x)) + x
