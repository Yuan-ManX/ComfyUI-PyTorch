# Code from: https://github.com/Alpha-VLLM/Lumina-Image-2.0/blob/main/models/model.py
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import common_dit

from modules.diffusionmodules.mmdit import TimestepEmbedder
from modules.attention import optimized_attention_masked
from flux.layers import EmbedND


def modulate(x, scale):
    """
    对输入张量 `x` 进行调制，通过将其与缩放因子 `scale` 相乘来实现。

    调制公式:
        modulated_x = x * (1 + scale.unsqueeze(1))

    这意味着每个元素 `x[i]` 都会被 `1 + scale[i]` 缩放。

    参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, ..., dim)。
        scale (torch.Tensor): 缩放因子张量，形状为 (batch_size, dim_scale)。

    返回:
        torch.Tensor: 调制后的张量，形状为 (batch_size, ..., dim)。
    """
    return x * (1 + scale.unsqueeze(1))


#############################################################################
#                               Core NextDiT Model                              #
#############################################################################


class JointAttention(nn.Module):
    """
    JointAttention 类实现了多头注意力机制，用于捕捉输入数据中的复杂依赖关系。
    该类支持使用 GQA（Grouped Query Attention）技术，通过设置 `n_kv_heads` 参数来指定键和值的头数。

    参数:
        dim (int): 输入特征的维度。
        n_heads (int): 多头注意力的头数。
        n_kv_heads (Optional[int]): 键和值的头数。如果为 None，则默认为 `n_heads`，表示不使用 GQA。
        qk_norm (bool): 是否对查询和键进行归一化，默认为 False。
        operation_settings (Dict): 操作设置，包含 `operations` 对象和其他配置参数，默认为空字典。
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int],
        qk_norm: bool,
        operation_settings={},
    ):
        super().__init__()
        # 设置键和值的头数，如果未指定，则默认为 `n_heads`
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        # 本地多头注意力的头数
        self.n_local_heads = n_heads
        # 本地键和值的头数
        self.n_local_kv_heads = self.n_kv_heads
        # 重复因子，用于重复键和值张量
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头的维度
        self.head_dim = dim // n_heads

        # 初始化 QKV 线性变换层
        self.qkv = operation_settings.get("operations").Linear(
            dim,
            (n_heads + self.n_kv_heads + self.n_kv_heads) * self.head_dim, # Q, K, V 的总维度
            bias=False, # 不使用偏置
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )
        # 初始化输出线性变换层
        self.out = operation_settings.get("operations").Linear(
            n_heads * self.head_dim,
            dim,
            bias=False,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )

        if qk_norm:
            # 如果需要对查询和键进行归一化，则初始化 RMSNorm 层
            self.q_norm = operation_settings.get("operations").RMSNorm(self.head_dim, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
            self.k_norm = operation_settings.get("operations").RMSNorm(self.head_dim, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        else:
            self.q_norm = self.k_norm = nn.Identity()

    @staticmethod
    def apply_rotary_emb(
        x_in: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        """
        应用旋转位置编码（RoPE）到输入张量 `x_in`，使用预计算的正弦和余弦频率张量 `freqs_cis`。

        该函数将旋转位置编码应用到给定的查询 `xq` 和键 `xk` 张量中。输入张量被重塑为复数形式，频率张量被重塑以实现广播兼容。最终结果包含旋转位置编码，并作为实数张量返回。

        参数:
            x_in (torch.Tensor): 要应用旋转位置编码的查询或键张量。
            freqs_cis (torch.Tensor): 预计算的正弦和余弦频率张量。

        返回:
            torch.Tensor: 应用旋转位置编码后的张量。
        """

        # 重塑张量以适应复数表示
        t_ = x_in.reshape(*x_in.shape[:-1], -1, 1, 2)
        # 应用旋转位置编码
        t_out = freqs_cis[..., 0] * t_[..., 0] + freqs_cis[..., 1] * t_[..., 1]
        # 重塑回原始形状
        return t_out.reshape(*x_in.shape)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播过程，计算多头注意力输出。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seqlen, dim)。
            x_mask (torch.Tensor): 输入掩码张量，形状为 (batch_size, seqlen)。
            freqs_cis (torch.Tensor): 预计算的正弦和余弦频率张量。

        返回:
            torch.Tensor: 注意力输出张量，形状为 (batch_size, seqlen, dim)。
        """
        # 获取批量大小、序列长度和特征维度
        bsz, seqlen, _ = x.shape

        # 分割 Q, K, V 张量
        xq, xk, xv = torch.split(
            self.qkv(x),
            [
                self.n_local_heads * self.head_dim,  # 查询的维度
                self.n_local_kv_heads * self.head_dim, # 键的维度
                self.n_local_kv_heads * self.head_dim, # 值的维度
            ],
            dim=-1,
        )

        # 重塑查询张量
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        # 重塑键张量
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        # 重塑值张量
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 对查询进行归一化
        xq = self.q_norm(xq)
        # 对键进行归一化
        xk = self.k_norm(xk)

        # 应用旋转位置编码
        xq = JointAttention.apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = JointAttention.apply_rotary_emb(xk, freqs_cis=freqs_cis)

        # 计算重复因子
        n_rep = self.n_local_heads // self.n_local_kv_heads
        if n_rep >= 1:
            # 如果重复因子大于等于1，则重复键和值张量以匹配查询的头数
            xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
        # 计算注意力输出，使用优化的掩码注意力函数
        output = optimized_attention_masked(xq.movedim(1, 2), xk.movedim(1, 2), xv.movedim(1, 2), self.n_local_heads, x_mask, skip_reshape=True)

        # 通过输出线性层进行投影并返回
        return self.out(output)


class FeedForward(nn.Module):
    """
    FeedForward 类实现了一个前馈神经网络（Feed-Forward Network, FFN），通常用于 Transformer 模型中。
    该 FFN 包含两个线性层，并使用 SiLU（也称为 Swish）激活函数进行非线性变换。

    参数:
        dim (int): 输入和输出的维度。
        hidden_dim (int): 隐藏层的维度。
        multiple_of (int): 确保隐藏层维度是此值的倍数，用于对齐或优化内存。
        ffn_dim_multiplier (float, optional): 隐藏层维度的自定义乘数，默认为 None。
        operation_settings (Dict, optional): 操作设置，包含 `operations` 对象和其他配置参数，默认为空字典。
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        operation_settings={},
    ):
        super().__init__()

        # 自定义隐藏层维度乘数
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)

        # 调整隐藏层维度，使其是 `multiple_of` 的倍数
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # 第一个线性层，将输入维度映射到隐藏层维度
        self.w1 = operation_settings.get("operations").Linear(
            dim,
            hidden_dim,
            bias=False,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )

        # 第二个线性层，将隐藏层维度映射回输入维度
        self.w2 = operation_settings.get("operations").Linear(
            hidden_dim,
            dim,
            bias=False,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )

        # 第三个线性层，与第一个线性层相同，用于处理输入的另一个分支
        self.w3 = operation_settings.get("operations").Linear(
            dim,
            hidden_dim,
            bias=False,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )

    # 使用 torch.compile 编译优化（可选）
    # @torch.compile
    def _forward_silu_gating(self, x1, x3):
        """
        使用 SiLU 激活函数对输入进行非线性变换，并进行门控。

        参数:
            x1 (torch.Tensor): 第一个输入张量。
            x3 (torch.Tensor): 第二个输入张量。

        返回:
            torch.Tensor: 经过 SiLU 激活和门控后的张量。
        """
        return F.silu(x1) * x3

    def forward(self, x):
        """
        前向传播过程，通过前馈神经网络处理输入。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: FFN 处理后的输出张量。
        """
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class JointTransformerBlock(nn.Module):
    """
    JointTransformerBlock 类实现了一个 Transformer 块，结合了多头自注意力机制和前馈神经网络（FFN）。
    该块支持使用 GQA（Grouped Query Attention）技术，并通过自适应层归一化（adaLN）进行调制。

    参数:
        layer_id (int): Transformer 块的层编号。
        dim (int): 输入特征的维度。
        n_heads (int): 多头注意力的头数。
        n_kv_heads (int): 键和值的多头注意力的头数。如果使用 GQA，则可以不同；否则，默认为 `n_heads`。
        multiple_of (int): 确保隐藏层维度是此值的倍数，用于对齐或优化内存。
        ffn_dim_multiplier (float): FFN 隐藏层维度的乘数。
        norm_eps (float): RMSNorm 的小常数，防止除零错误。
        qk_norm (bool): 是否对查询和键进行归一化，默认为 False。
        modulation (bool, optional): 是否启用自适应层归一化调制，默认为 True。
        operation_settings (Dict, optional): 操作设置，包含 `operations` 对象和其他配置参数，默认为空字典。
    """
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
        operation_settings={},
    ) -> None:
        
        super().__init__()

        # 输入特征的维度
        self.dim = dim
        # 每个头的维度
        self.head_dim = dim // n_heads

        # 初始化多头注意力机制
        self.attention = JointAttention(dim, n_heads, n_kv_heads, qk_norm, operation_settings=operation_settings)
        
        # 初始化前馈神经网络
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,  # 隐藏层维度为 4 * dim
            multiple_of=multiple_of,  # 隐藏层维度的倍数
            ffn_dim_multiplier=ffn_dim_multiplier,  # FFN 隐藏层维度的乘数
            operation_settings=operation_settings,
        )
        # Transformer 块的层编号
        self.layer_id = layer_id

        # 初始化 RMSNorm 层
        self.attention_norm1 = operation_settings.get("operations").RMSNorm(dim, eps=norm_eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.ffn_norm1 = operation_settings.get("operations").RMSNorm(dim, eps=norm_eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))

        self.attention_norm2 = operation_settings.get("operations").RMSNorm(dim, eps=norm_eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.ffn_norm2 = operation_settings.get("operations").RMSNorm(dim, eps=norm_eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))

        # 是否启用自适应层归一化调制
        self.modulation = modulation
        if modulation:
            # 如果启用自适应层归一化调制，则初始化 adaLN 调制模块
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                operation_settings.get("operations").Linear(
                    min(dim, 1024),  # 输入维度为 dim 和 1024 的较小值
                    4 * dim,  # 输出维度为 4 * dim
                    bias=True,
                    device=operation_settings.get("device"),
                    dtype=operation_settings.get("dtype"),
                ),
            )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor]=None,
    ):
        """
        前向传播过程，通过 Transformer 块处理输入数据。

        参数:
            x (torch.Tensor): 输入张量。
            x_mask (torch.Tensor): 输入掩码张量。
            freqs_cis (torch.Tensor): 预计算的正弦和余弦频率张量。
            adaln_input (torch.Tensor, optional): 自适应层归一化输入，默认为 None。

        返回:
            torch.Tensor: Transformer 块处理后的输出张量。
        """
        if self.modulation:
            # 如果启用自适应层归一化调制，则断言 adaLN 输入不为 None
            assert adaln_input is not None
            # 分割自适应层归一化调制参数
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)

            # 应用自适应层归一化调制到注意力部分
            x = x + gate_msa.unsqueeze(1).tanh() * self.attention_norm2(
                self.attention(
                    modulate(self.attention_norm1(x), scale_msa), # 调制注意力输入
                    x_mask,
                    freqs_cis,
                )
            )

            # 应用自适应层归一化调制到前馈神经网络部分
            x = x + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(
                self.feed_forward(
                    modulate(self.ffn_norm1(x), scale_mlp),  # 调制 FFN 输入
                )
            )
        else:
            # 如果未启用自适应层归一化调制，则断言 adaLN 输入为 None
            assert adaln_input is None
            # 应用标准层归一化到注意力部分
            x = x + self.attention_norm2(
                self.attention(
                    self.attention_norm1(x),
                    x_mask,
                    freqs_cis,
                )
            )
            # 应用标准层归一化到前馈神经网络部分
            x = x + self.ffn_norm2(
                self.feed_forward(
                    self.ffn_norm1(x),
                )
            )
        # 返回处理后的输出张量
        return x


class FinalLayer(nn.Module):
    """
    NextDiT 模型中的最后一层，用于将 Transformer 的输出转换为最终的输出张量。
    该层包含一个层归一化、一个线性层以及自适应层归一化（adaLN）调制模块。

    参数:
        hidden_size (int): 隐藏层的大小。
        patch_size (int): 图像小块的大小。
        out_channels (int): 输出通道数。
        operation_settings (Dict, optional): 操作设置，包含 `operations` 对象和其他配置参数，默认为空字典。
    """

    def __init__(self, hidden_size, patch_size, out_channels, operation_settings={}):
        super().__init__()

        # 初始化层归一化层，不使用元素级仿射变换
        self.norm_final = operation_settings.get("operations").LayerNorm(
            hidden_size,
            elementwise_affine=False,  # 不使用元素级仿射变换
            eps=1e-6,  # 小常数，防止除零错误
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )

        # 初始化线性层，将隐藏层大小映射到 patch_size * patch_size * out_channels
        self.linear = operation_settings.get("operations").Linear(
            hidden_size,
            patch_size * patch_size * out_channels,   # 输出维度
            bias=True,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )

        # 初始化自适应层归一化（adaLN）调制模块
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            operation_settings.get("operations").Linear(
                min(hidden_size, 1024),  # 输入维度为 hidden_size 和 1024 的较小值
                hidden_size, # 输出维度为 hidden_size
                bias=True,
                device=operation_settings.get("device"),
                dtype=operation_settings.get("dtype"),
            ),
        )

    def forward(self, x, c):
        """
        前向传播过程，将输入张量通过层归一化、自适应层归一化调制和线性层进行处理。

        参数:
            x (torch.Tensor): 输入张量。
            c (torch.Tensor): 条件张量，用于自适应层归一化调制。

        返回:
            torch.Tensor: 处理后的输出张量。
        """
        # 通过 adaLN 调制模块计算缩放因子
        scale = self.adaLN_modulation(c)
        # 对归一化后的输入进行调制
        x = modulate(self.norm_final(x), scale)
        # 通过线性层进行投影
        x = self.linear(x)
        return x


class NextDiT(nn.Module):
    """
    NextDiT 模型是一个基于 Transformer 的扩散模型，用于生成图像。
    该模型结合了时间步嵌入、文本嵌入、图像小块嵌入、旋转位置编码（RoPE）以及多头自注意力机制。

    参数:
        patch_size (int, optional): 图像小块的大小，默认为2。
        in_channels (int, optional): 输入图像的通道数，默认为4。
        dim (int, optional): Transformer 的特征维度，默认为4096。
        n_layers (int, optional): Transformer 层的数量，默认为32。
        n_refiner_layers (int, optional): 精炼 Transformer 层的数量，默认为2。
        n_heads (int, optional): 多头注意力的头数，默认为32。
        n_kv_heads (Optional[int], optional): 键和值的多头注意力的头数，默认为 None。
        multiple_of (int, optional): 确保隐藏层维度是此值的倍数，默认为256。
        ffn_dim_multiplier (float, optional): FFN 隐藏层维度的乘数，默认为 None。
        norm_eps (float, optional): RMSNorm 的小常数，默认为1e-5。
        qk_norm (bool, optional): 是否对查询和键进行归一化，默认为 False。
        cap_feat_dim (int, optional): 捕获特征维度，默认为5120。
        axes_dims (List[int], optional): RoPE 的轴维度，默认为 (16, 56, 56)。
        axes_lens (List[int], optional): RoPE 的轴长度，默认为 (1, 512, 512)。
        image_model (object, optional): 图像模型，默认为 None。
        device (torch.device, optional): 设备（CPU 或 GPU），默认为 None。
        dtype (torch.dtype, optional): 数据类型，默认为 None。
        operations (object, optional): 自定义操作对象，默认为 None。
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 4,
        dim: int = 4096,
        n_layers: int = 32,
        n_refiner_layers: int = 2,
        n_heads: int = 32,
        n_kv_heads: Optional[int] = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        qk_norm: bool = False,
        cap_feat_dim: int = 5120,
        axes_dims: List[int] = (16, 56, 56),
        axes_lens: List[int] = (1, 512, 512),
        image_model=None,
        device=None,
        dtype=None,
        operations=None,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        operation_settings = {"operations": operations, "device": device, "dtype": dtype}

        # 输入图像的通道数
        self.in_channels = in_channels
        # 输出图像的通道数
        self.out_channels = in_channels
        # 图像小块的大小
        self.patch_size = patch_size

        # 图像小块嵌入器，将输入图像小块映射到 Transformer 的特征维度
        self.x_embedder = operation_settings.get("operations").Linear(
            in_features=patch_size * patch_size * in_channels,  # 输入特征维度
            out_features=dim,  # 输出特征维度
            bias=True,  # 使用偏置
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )

        # 噪声精炼模块，包含多个 Transformer 块
        self.noise_refiner = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    modulation=True,  # 启用自适应层归一化调制
                    operation_settings=operation_settings,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )

        # 上下文精炼模块，包含多个 Transformer 块
        self.context_refiner = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    modulation=False,  # 不启用自适应层归一化调制
                    operation_settings=operation_settings,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )

        # 时间步嵌入器
        self.t_embedder = TimestepEmbedder(min(dim, 1024), **operation_settings)
        # 文本嵌入器
        self.cap_embedder = nn.Sequential(
            operation_settings.get("operations").RMSNorm(cap_feat_dim, eps=norm_eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")),
            operation_settings.get("operations").Linear(
                cap_feat_dim,
                dim,
                bias=True,
                device=operation_settings.get("device"),
                dtype=operation_settings.get("dtype"),
            ),
        )

        # Transformer 层列表
        self.layers = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    operation_settings=operation_settings,
                )
                for layer_id in range(n_layers)
            ]
        )

        # 最终层归一化
        self.norm_final = operation_settings.get("operations").RMSNorm(dim, eps=norm_eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        # 最终输出层
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels, operation_settings=operation_settings)

        # 确保维度匹配
        assert (dim // n_heads) == sum(axes_dims)
        # RoPE 的轴维度
        self.axes_dims = axes_dims
        # RoPE 的轴长度
        self.axes_lens = axes_lens
        # RoPE 嵌入器
        self.rope_embedder = EmbedND(dim=dim // n_heads, theta=10000.0, axes_dim=axes_dims)
        # Transformer 的特征维度
        self.dim = dim
        # 多头注意力的头数
        self.n_heads = n_heads

    def unpatchify(
        self, x: torch.Tensor, img_size: List[Tuple[int, int]], cap_size: List[int], return_tensor=False
    ) -> List[torch.Tensor]:
        """
        将小块图像张量重新排列为原始图像张量。

        参数:
            x (torch.Tensor): 输入的小块图像张量，形状为 (N, T, patch_size**2 * C)。
            img_size (List[Tuple[int, int]]): 每个图像的尺寸列表，元素为 (高度, 宽度)。
            cap_size (List[int]): 每个图像的起始位置列表。
            return_tensor (bool, optional): 是否返回张量，默认为 False。

        返回:
            List[torch.Tensor]: 重新排列后的图像张量列表。
        """
        # 小块的高度和宽度
        pH = pW = self.patch_size
        # 初始化图像列表
        imgs = []
        for i in range(x.size(0)):
            # 获取当前图像的尺寸
            H, W = img_size[i]
            # 获取当前图像的起始位置
            begin = cap_size[i]
            # 计算结束位置
            end = begin + (H // pH) * (W // pW)
            # 重新排列小块图像为原始图像形状
            imgs.append(
                x[i][begin:end]
                .view(H // pH, W // pW, pH, pW, self.out_channels)
                .permute(4, 0, 2, 1, 3)
                .flatten(3, 4)
                .flatten(1, 2)
            )

        if return_tensor:
            # 如果需要返回张量，则将列表转换为张量
            imgs = torch.stack(imgs, dim=0)
        # 返回重新排列后的图像列表
        return imgs

    def patchify_and_embed(
        self, x: List[torch.Tensor] | torch.Tensor, cap_feats: torch.Tensor, cap_mask: torch.Tensor, t: torch.Tensor, num_tokens
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]], List[int], torch.Tensor]:
        """
        将输入图像分割成小块并进行嵌入，同时处理文本特征。

        参数:
            x (List[torch.Tensor] | torch.Tensor): 输入图像列表或张量。
            cap_feats (torch.Tensor): 文本特征张量。
            cap_mask (torch.Tensor): 文本掩码张量。
            t (torch.Tensor): 时间步张量。
            num_tokens (int): 文本的最大 token 数量。

        返回:
            Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]], List[int], torch.Tensor]:
                - x (torch.Tensor): 分割并嵌入后的图像张量。
                - mask (torch.Tensor): 图像掩码张量。
                - img_sizes (List[Tuple[int, int]]): 每个图像的尺寸列表。
                - cap_size (List[int]): 每个图像的文本长度列表。
                - freqs_cis (torch.Tensor): 旋转位置编码张量。
        """
        # 获取批量大小
        bsz = len(x)
        # 获取图像小块的大小
        pH = pW = self.patch_size
        device = x[0].device
        dtype = x[0].dtype

        if cap_mask is not None:
            # 计算有效的文本长度
            l_effective_cap_len = cap_mask.sum(dim=1).tolist()
        else:
            # 如果没有掩码，则使用最大 token 数量
            l_effective_cap_len = [num_tokens] * bsz

        if cap_mask is not None and not torch.is_floating_point(cap_mask):
            # 处理掩码张量
            cap_mask = (cap_mask - 1).to(dtype) * torch.finfo(dtype).max

        # 获取每个图像的尺寸
        img_sizes = [(img.size(1), img.size(2)) for img in x]
        # 计算每个图像的有效小块数量
        l_effective_img_len = [(H // pH) * (W // pW) for (H, W) in img_sizes]

        # 计算最大序列长度
        max_seq_len = max(
            (cap_len+img_len for cap_len, img_len in zip(l_effective_cap_len, l_effective_img_len))
        )
        # 获取最大的文本长度
        max_cap_len = max(l_effective_cap_len)
        # 获取最大的图像小块数量
        max_img_len = max(l_effective_img_len)

        # 初始化位置 ID 张量
        position_ids = torch.zeros(bsz, max_seq_len, 3, dtype=torch.int32, device=device)

        for i in range(bsz):
            # 获取当前批次的文本长度
            cap_len = l_effective_cap_len[i]
            # 获取当前批次的图像小块数量
            img_len = l_effective_img_len[i]
            # 获取当前图像的尺寸
            H, W = img_sizes[i]
            # 计算图像的高度和宽度上的小块数量
            H_tokens, W_tokens = H // pH, W // pW
            # 确保小块数量匹配
            assert H_tokens * W_tokens == img_len

            # 设置文本位置 ID
            position_ids[i, :cap_len, 0] = torch.arange(cap_len, dtype=torch.int32, device=device)
            # 设置图像位置 ID
            position_ids[i, cap_len:cap_len+img_len, 0] = cap_len
            # 生成行 ID
            row_ids = torch.arange(H_tokens, dtype=torch.int32, device=device).view(-1, 1).repeat(1, W_tokens).flatten()
            # 生成列 ID
            col_ids = torch.arange(W_tokens, dtype=torch.int32, device=device).view(1, -1).repeat(H_tokens, 1).flatten()
            # 设置行 ID
            position_ids[i, cap_len:cap_len+img_len, 1] = row_ids
            # 设置列 ID
            position_ids[i, cap_len:cap_len+img_len, 2] = col_ids

        # 计算旋转位置编码
        freqs_cis = self.rope_embedder(position_ids).movedim(1, 2).to(dtype)

        # 分别构建文本和图像的旋转位置编码
        cap_freqs_cis_shape = list(freqs_cis.shape)
        # cap_freqs_cis_shape[1] = max_cap_len
        # 设置文本的旋转位置编码维度
        cap_freqs_cis_shape[1] = cap_feats.shape[1]
        # 初始化文本旋转位置编码
        cap_freqs_cis = torch.zeros(*cap_freqs_cis_shape, device=device, dtype=freqs_cis.dtype)

        img_freqs_cis_shape = list(freqs_cis.shape)
        # 设置图像的旋转位置编码维度
        img_freqs_cis_shape[1] = max_img_len
        # 初始化图像旋转位置编码
        img_freqs_cis = torch.zeros(*img_freqs_cis_shape, device=device, dtype=freqs_cis.dtype)

        for i in range(bsz):
            # 获取当前批次的文本长度
            cap_len = l_effective_cap_len[i]
            # 获取当前批次的图像小块数量
            img_len = l_effective_img_len[i]
            # 设置文本旋转位置编码
            cap_freqs_cis[i, :cap_len] = freqs_cis[i, :cap_len]
            # 设置图像旋转位置编码
            img_freqs_cis[i, :img_len] = freqs_cis[i, cap_len:cap_len+img_len]

        # 精炼上下文
        for layer in self.context_refiner:
            # 通过上下文精炼模块处理文本特征
            cap_feats = layer(cap_feats, cap_mask, cap_freqs_cis)

        # 精炼图像
        flat_x = []  # 初始化图像列表
        for i in range(bsz):
            # 获取当前图像
            img = x[i]
            # 获取图像的通道数、高度和宽度
            C, H, W = img.size()
            # 分割图像为小块并展平
            img = img.view(C, H // pH, pH, W // pW, pW).permute(1, 3, 2, 4, 0).flatten(2).flatten(0, 1)
            # 添加到列表中
            flat_x.append(img)
        # 更新图像列表
        x = flat_x
        # 初始化填充后的图像嵌入
        padded_img_embed = torch.zeros(bsz, max_img_len, x[0].shape[-1], device=device, dtype=x[0].dtype)
        # 初始化图像掩码
        padded_img_mask = torch.zeros(bsz, max_img_len, dtype=dtype, device=device)
        for i in range(bsz):
            # 填充图像嵌入
            padded_img_embed[i, :l_effective_img_len[i]] = x[i]
            # 设置掩码
            padded_img_mask[i, l_effective_img_len[i]:] = -torch.finfo(dtype).max

        # 对填充后的图像嵌入进行嵌入
        padded_img_embed = self.x_embedder(padded_img_embed)
        # 增加维度以匹配注意力机制
        padded_img_mask = padded_img_mask.unsqueeze(1)
        for layer in self.noise_refiner:
            # 通过噪声精炼模块处理图像嵌入
            padded_img_embed = layer(padded_img_embed, padded_img_mask, img_freqs_cis, t)

        if cap_mask is not None:
            # 初始化掩码张量
            mask = torch.zeros(bsz, max_seq_len, dtype=dtype, device=device)
            # 设置文本掩码
            mask[:, :max_cap_len] = cap_mask[:, :max_cap_len]
        else:
            mask = None

        # 初始化填充后的完整嵌入
        padded_full_embed = torch.zeros(bsz, max_seq_len, self.dim, device=device, dtype=x[0].dtype)
        for i in range(bsz):
            # 获取当前批次的文本长度
            cap_len = l_effective_cap_len[i]
            # 获取当前批次的图像小块数量
            img_len = l_effective_img_len[i]

            # 填充文本嵌入
            padded_full_embed[i, :cap_len] = cap_feats[i, :cap_len]
            # 填充图像嵌入
            padded_full_embed[i, cap_len:cap_len+img_len] = padded_img_embed[i, :img_len]

        # 返回填充后的完整嵌入、掩码、图像尺寸、文本长度和旋转位置编码
        return padded_full_embed, mask, img_sizes, l_effective_cap_len, freqs_cis

    # def forward(self, x, t, cap_feats, cap_mask):
    def forward(self, x, timesteps, context, num_tokens, attention_mask=None, **kwargs):
        """
        前向传播过程，处理输入图像和时间步。

        参数:
            x (torch.Tensor): 输入图像张量。
            timesteps (torch.Tensor): 时间步张量。
            context (torch.Tensor): 上下文张量。
            num_tokens (int): 文本的最大 token 数量。
            attention_mask (torch.Tensor, optional): 注意力掩码，默认为 None。

        返回:
            torch.Tensor: 处理后的输出张量。
        """
        # 计算时间步的补数
        t = 1.0 - timesteps
        # 获取上下文作为文本特征
        cap_feats = context
        # 获取注意力掩码
        cap_mask = attention_mask
        # 获取输入图像的批量大小、通道数、高度和宽度
        bs, c, h, w = x.shape
        # 对输入图像进行填充以适应小块大小
        x = common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
        # 通过时间步嵌入器进行嵌入
        t = self.t_embedder(t, dtype=x.dtype)  # (N, D)
        # 设置自适应层归一化输入
        adaln_input = t

        # 通过文本嵌入器进行嵌入
        cap_feats = self.cap_embedder(cap_feats)  # (N, L, D)  # todo check if able to batchify w.o. redundant compute

        # 判断输入是否为张量
        x_is_tensor = isinstance(x, torch.Tensor)
        # 分割图像并进行嵌入
        x, mask, img_size, cap_size, freqs_cis = self.patchify_and_embed(x, cap_feats, cap_mask, t, num_tokens)
        # 将旋转位置编码移动到输入设备
        freqs_cis = freqs_cis.to(x.device)

        for layer in self.layers:
            # 通过 Transformer 层处理输入
            x = layer(x, mask, freqs_cis, adaln_input)

        # 通过最终层进行输出投影
        x = self.final_layer(x, adaln_input)
        # 重新排列图像并裁剪到原始大小
        x = self.unpatchify(x, img_size, cap_size, return_tensor=x_is_tensor)[:,:,:h,:w]

        return -x

