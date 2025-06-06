import torch
from einops import rearrange
from torch import Tensor

from modules.attention import optimized_attention
import model_management


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, mask=None) -> Tensor:
    """
    应用多头自注意力机制。

    参数:
        q (torch.Tensor): 查询张量，形状为 (batch_size, num_heads, seq_len, head_dim)。
        k (torch.Tensor): 键张量，形状为 (batch_size, num_heads, seq_len, head_dim)。
        v (torch.Tensor): 值张量，形状为 (batch_size, num_heads, seq_len, head_dim)。
        pe (torch.Tensor): 位置嵌入张量，形状为 (batch_size, seq_len, dim)。
        mask (torch.Tensor, 可选): 注意力掩码，默认为 None。

    返回:
        torch.Tensor: 注意力机制的输出，形状为 (batch_size, num_heads, seq_len, head_dim)。
    """
    # 保存查询张量的原始形状
    q_shape = q.shape
    # 保存键张量的原始形状
    k_shape = k.shape

    if pe is not None:
        # 如果提供了位置嵌入，则将其应用于查询和键
        # 将查询和键转换为与位置嵌入相同的数据类型，并重塑为 (batch_size, num_heads, seq_len, 1, 2)
        q = q.to(dtype=pe.dtype).reshape(*q.shape[:-1], -1, 1, 2)
        k = k.to(dtype=pe.dtype).reshape(*k.shape[:-1], -1, 1, 2)

        # 应用位置嵌入到查询和键上
        q = (pe[..., 0] * q[..., 0] + pe[..., 1] * q[..., 1]).reshape(*q_shape).type_as(v)
        k = (pe[..., 0] * k[..., 0] + pe[..., 1] * k[..., 1]).reshape(*k_shape).type_as(v)

    # 获取头的数量
    heads = q.shape[1]
    # 调用优化的注意力函数，传入查询、键、值、头的数量、是否跳过重塑标志和掩码
    x = optimized_attention(q, k, v, heads, skip_reshape=True, mask=mask)
    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    """
    应用旋转位置编码（RoPE）。

    参数:
        pos (torch.Tensor): 位置张量，形状为 (..., seq_len, dim)。
        dim (int): 位置嵌入的维度，必须为偶数。
        theta (int): 旋转角度参数。

    返回:
        torch.Tensor: 应用了 RoPE 的张量，形状为 (..., seq_len, dim, 2, 2)。
    """
    # 确保维度为偶数
    assert dim % 2 == 0
    # 根据设备类型选择计算设备，避免在某些设备上计算 RoPE
    if model_management.is_device_mps(pos.device) or model_management.is_intel_xpu() or model_management.is_directml_enabled():
        device = torch.device("cpu")
    else:
        device = pos.device

    # 计算缩放因子
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim//2, dtype=torch.float64, device=device)
    # 计算 omega 值
    omega = 1.0 / (theta**scale)
    # 计算 RoPE
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    # 构建旋转矩阵
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    # 重塑张量以匹配 (..., seq_len, dim, 2, 2) 的形状
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor):
    """
    应用 RoPE 到查询和键张量。

    参数:
        xq (torch.Tensor): 查询张量，形状为 (..., seq_len, dim)。
        xk (torch.Tensor): 键张量，形状为 (..., seq_len, dim)。
        freqs_cis (torch.Tensor): 预计算的 RoPE 张量，形状为 (..., seq_len, dim, 2)。

    返回:
        tuple: 应用了 RoPE 的查询和键张量，形状均为 (..., seq_len, dim)。
    """
    # 将查询和键转换为与 RoPE 相同的数据类型，并重塑为 (..., seq_len, dim/2, 2)
    xq_ = xq.to(dtype=freqs_cis.dtype).reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.to(dtype=freqs_cis.dtype).reshape(*xk.shape[:-1], -1, 1, 2)

    # 应用 RoPE 到查询和键
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    
    # 重塑回原始形状
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

