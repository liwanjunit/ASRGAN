import torch
import torch.nn as nn


class Attention(nn.Module):
	def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
		super().__init__()
		self.num_heads = num_heads
		head_dim = dim // num_heads
		# NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
		self.scale = qk_scale or head_dim ** -0.5

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

	def forward(self, x):
		# print(x.shape)
		# print(self.qkv(x).shape)
		B, N, C = x.shape
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

		attn = (q @ k.transpose(-2, -1)) * self.scale
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)

		x = (attn @ v).transpose(1, 2).reshape(B, N, C)
		x = self.proj(x)
		x = self.proj_drop(x)
		# print("x after -->", x.shape)
		return x


class CrossAttention(nn.Module):
	def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
		super().__init__()
		self.num_heads = num_heads
		self.dim = dim
		self.out_dim = out_dim
		head_dim = out_dim // num_heads
		# NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
		self.scale = qk_scale or head_dim ** -0.5

		self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
		self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
		self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)

		self.proj = nn.Linear(out_dim, out_dim)
		self.proj_drop = nn.Dropout(proj_drop)

	def forward(self, q, v):
		B, N, _ = q.shape
		C = self.out_dim
		k = v
		NK = k.size(1)

		q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
		k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
		v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

		attn = (q @ k.transpose(-2, -1)) * self.scale
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)

		x = (attn @ v).transpose(1, 2).reshape(B, N, C)
		x = self.proj(x)
		x = self.proj_drop(x)
		return x
