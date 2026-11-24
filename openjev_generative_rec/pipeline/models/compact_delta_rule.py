"""Low-memory PyTorch gated delta rule for the local Qwen pilot."""

import torch
from torch.nn import functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm


def compact_gated_delta_rule(query, key, value, g, beta, *, initial_state=None,
                             output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
    dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query, key = (l2norm(tensor, dim=-1, eps=1e-6) for tensor in (query, key))
    query, key, value, beta, g = (tensor.transpose(1, 2).contiguous().float()
                                   for tensor in (query, key, value, beta, g))
    batch, heads, length, key_dim = key.shape
    value_dim = value.shape[-1]
    chunk = min(64, length)
    padding = (-length) % chunk
    query, key, value = (F.pad(tensor, (0, 0, 0, padding)).unflatten(2, (-1, chunk))
                         for tensor in (query, key, value))
    query = query * (key_dim ** -0.5)
    beta = F.pad(beta, (0, padding)).unflatten(2, (-1, chunk)).unsqueeze(-1)
    g = F.pad(g, (0, padding)).unflatten(2, (-1, chunk)).cumsum(dim=-1)
    delta = g.unsqueeze(-1) - g.unsqueeze(-2)
    decay = delta.tril().exp().tril()
    beta_keys = beta * key
    lower = ((beta_keys @ key.transpose(-1, -2)) * decay).tril(-1)

    # The reference row recurrence applies (I + lower)^-1 to both right-hand sides.
    rhs = torch.cat((beta * value, beta_keys * g.exp().unsqueeze(-1)), dim=-1)
    solved = torch.linalg.solve_triangular(lower, rhs, upper=False, unitriangular=True)
    values, keys = solved.split((value_dim, key_dim), dim=-1)
    state = (value.new_zeros(batch, heads, key_dim, value_dim)
             if initial_state is None else initial_state.to(value))
    outputs = []
    for q_i, k_i, v_i, g_i, key_i, decay_i in zip(
        query.unbind(2), key.unbind(2), values.unbind(2), g.unbind(2), keys.unbind(2), decay.unbind(2), strict=True,
    ):
        v_new = v_i - key_i @ state
        local = (q_i @ k_i.transpose(-1, -2)) * decay_i
        outputs.append((q_i * g_i.exp().unsqueeze(-1)) @ state + local @ v_new)
        end = g_i[..., -1:]
        state = state * end.exp().unsqueeze(-1) + (k_i * (end - g_i).exp().unsqueeze(-1)).transpose(-1, -2) @ v_new
    hidden = torch.cat(outputs, dim=2)[:, :, :length].transpose(1, 2).contiguous().to(dtype)
    return hidden, state if output_final_state else None
