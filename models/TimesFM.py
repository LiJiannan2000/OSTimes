from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


def get_large_negative_number(dtype: torch.dtype) -> torch.Tensor:
  """Returns a large negative value for the given dtype."""
  if dtype.is_floating_point:
    dtype_max = torch.finfo(dtype).max
  else:
    dtype_max = torch.iinfo(dtype).max
  return torch.tensor(-0.7 * dtype_max, dtype=dtype)


def causal_mask(input_t: torch.Tensor) -> torch.Tensor:
  """Computes and returns causal mask.

  Args:
      input_t: A torch.Tensor of shape [B, T, D].

  Returns:
      An attention_mask torch.Tensor of shape [1, 1, T, T]. Attention mask has
      already been converted to large negative values.
  """
  assert input_t.dtype.is_floating_point, input_t.dtype
  large_negative_number = get_large_negative_number(input_t.dtype)
  t = input_t.shape[1]
  col_idx = torch.arange(t).unsqueeze(0).repeat(t, 1)
  row_idx = torch.arange(t).unsqueeze(1).repeat(1, t)
  mask = (row_idx < col_idx).to(input_t.dtype) * large_negative_number
  return (mask.unsqueeze(0).unsqueeze(0).to(input_t.device)
         )  # Equivalent to jnp.newaxis


class RMSNorm(torch.nn.Module):
  """Pax rms norm in pytorch."""

  def __init__(
      self,
      dim: int,
      eps: float = 1e-6,
      add_unit_offset: bool = False,
  ):
    super().__init__()
    self.eps = eps
    self.add_unit_offset = add_unit_offset
    self.weight = nn.Parameter(torch.zeros(dim))

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x):
    output = self._norm(x.float())
    if self.add_unit_offset:
      output = output * (1 + self.weight.float())
    else:
      output = output * self.weight.float()
    return output.type_as(x)


class TransformerMLP(nn.Module):
  """Pax transformer MLP in pytorch."""

  def __init__(
      self,
      hidden_size: int,
      intermediate_size: int,
  ):
    super().__init__()
    self.gate_proj = nn.Linear(hidden_size, intermediate_size)
    self.down_proj = nn.Linear(intermediate_size, hidden_size)
    self.layer_norm = nn.LayerNorm(normalized_shape=hidden_size, eps=1e-6)

  def forward(self, x, paddings=None):
    gate_inp = self.layer_norm(x)
    gate = self.gate_proj(gate_inp)
    gate = F.relu(gate)
    outputs = self.down_proj(gate)
    if paddings is not None:
      outputs = outputs * (1.0 - paddings[:, :, None])
    return outputs + x


class TimesFMAttention(nn.Module):
  """Implements the attention used in TimesFM."""

  def __init__(
      self,
      hidden_size: int,
      num_heads: int,
      num_kv_heads: int,
      head_dim: int,
  ):
    super().__init__()

    self.num_heads = num_heads
    self.num_kv_heads = num_kv_heads

    assert self.num_heads % self.num_kv_heads == 0
    self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    self.hidden_size = hidden_size
    self.head_dim = head_dim

    self.q_size = self.num_heads * self.head_dim
    self.kv_size = self.num_kv_heads * self.head_dim
    self.scaling = nn.Parameter(
        torch.empty((self.head_dim,), dtype=torch.float32),)

    self.qkv_proj = nn.Linear(
        self.hidden_size,
        (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
    )
    self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

  def _per_dim_scaling(self, query: torch.Tensor) -> torch.Tensor:
    # [batch_size, n_local_heads, input_len, head_dim]
    r_softplus_0 = 1.442695041
    softplus_func = torch.nn.Softplus()
    scale = r_softplus_0 / math.sqrt(self.head_dim)
    scale = scale * softplus_func(self.scaling)
    return query * scale[None, None, None, :]

  def forward(
      self,
      hidden_states: torch.Tensor,
      mask: torch.Tensor,
      kv_write_indices: torch.Tensor | None = None,
      kv_cache = None,
  ) -> torch.Tensor:
    hidden_states_shape = hidden_states.shape
    assert len(hidden_states_shape) == 3

    batch_size, input_len, _ = hidden_states_shape

    qkv = self.qkv_proj(hidden_states).to(torch.float32)
    xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
    xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
    xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)
    xq = self._per_dim_scaling(xq)

    # Write new kv cache.
    # [batch_size, input_len, n_local_kv_heads, head_dim]
    if kv_cache is not None and kv_write_indices is not None:
      k_cache, v_cache = kv_cache
      k_cache.index_copy_(1, kv_write_indices, xk)
      v_cache.index_copy_(1, kv_write_indices, xv)

      key = k_cache
      value = v_cache
    else:
      key = xk
      value = xv
    if self.num_kv_heads != self.num_heads:
      # [batch_size, max_seq_len, n_local_heads, head_dim]
      key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
      value = torch.repeat_interleave(value, self.num_queries_per_kv, dim=2)

    # [batch_size, n_local_heads, input_len, head_dim]
    q = xq.transpose(1, 2)
    # [batch_size, n_local_heads, max_seq_len, head_dim]
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    # [batch_size, n_local_heads, input_len, max_seq_len]
    scores = torch.matmul(q, k.transpose(2, 3))
    # scores = scores + mask
    scores = F.softmax(scores.float(), dim=-1).type_as(q)

    # [batch_size, n_local_heads, input_len, head_dim]
    output = torch.matmul(scores, v)
    # return scores, output.transpose(1, 2).contiguous()

    # [batch_size, input_len, hidden_dim]
    output = output.transpose(1, 2).contiguous().view(batch_size, input_len, -1)
    output = self.o_proj(output)
    return scores, output


class TimesFMDecoderLayer(nn.Module):
  """Transformer layer."""

  def __init__(
      self,
      hidden_size: int,
      intermediate_size: int,
      num_heads: int,
      num_kv_heads: int,
      head_dim: int,
      rms_norm_eps: float = 1e-6,
  ):
    super().__init__()
    self.self_attn = TimesFMAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    self.mlp = TransformerMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, add_unit_offset=True)

  def forward(
      self,
      hidden_states: torch.Tensor,
      mask: torch.Tensor,
      paddings: torch.Tensor,
      kv_write_indices: torch.Tensor | None = None,
      kv_cache = None,
  ) -> torch.Tensor:
    # Self Attention
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    scores, hidden_states = self.self_attn(
        hidden_states=hidden_states,
        mask=mask,
        kv_write_indices=kv_write_indices,
        kv_cache=kv_cache,
    )
    hidden_states = residual + hidden_states

    # MLP
    hidden_states = self.mlp(hidden_states, paddings=paddings)

    return scores, hidden_states


class StackedDecoder(nn.Module):
  """Stacked transformer layer."""

  def __init__(
      self,
      hidden_size: int,
      intermediate_size: int,
      num_heads: int,
      num_kv_heads: int,
      head_dim: int,
      num_layers: int,
      rms_norm_eps: float = 1e-6,
  ):
    super().__init__()

    self.layers = nn.ModuleList()
    for _ in range(num_layers):
      self.layers.append(
          TimesFMDecoderLayer(
              hidden_size=hidden_size,
              intermediate_size=intermediate_size,
              num_heads=num_heads,
              num_kv_heads=num_kv_heads,
              head_dim=head_dim,
              rms_norm_eps=rms_norm_eps,
          ))

  def forward(
      self,
      hidden_states: torch.Tensor,  # (32, 4, 1280)
      kv_write_indices: torch.Tensor | None = None,
      kv_caches = None,
      paddings: torch.Tenso | None = None,
  ) -> torch.Tensor:
    # padding_mask = convert_paddings_to_mask(paddings, hidden_states.dtype)
    atten_mask = causal_mask(hidden_states)  # (1, 1, 4, 4)
    mask = atten_mask.expand(hidden_states.shape[0], 1, 512, 512)
    # mask = merge_masks(padding_mask, atten_mask)
    for i in range(len(self.layers)):
      layer = self.layers[i]
      kv_cache = kv_caches[i] if kv_caches is not None else None
      _, hidden_states = layer(
          hidden_states=hidden_states,
          mask=mask,
          paddings=paddings,
          kv_write_indices=kv_write_indices,
          kv_cache=kv_cache,
      )
    return hidden_states