from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import einops
from einops import repeat

from lightricks.model import TimestepEmbedding, Timesteps
import torch.nn.functional as F

from flux.math import apply_rope, rope
from flux.layers import LastLayer

from modules.attention import optimized_attention
import model_management
import common_dit


# Copied from https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/layers.py
class EmbedND(nn.Module):
    """
    EmbedND 类用于对多维输入进行嵌入处理。
    它将输入的每个轴嵌入到一个高维空间中，并通过旋转位置编码（RoPE）增强位置信息。

    参数:
        theta (int): 用于旋转位置编码的旋转角度参数。
        axes_dim (List[int]): 每个轴的维度列表，用于确定每个轴的嵌入维度。
    """
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        # 旋转位置编码的旋转角度参数
        self.theta = theta
        # 每个轴的维度列表
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播过程，对输入的多维 ID 进行嵌入处理。

        参数:
            ids (torch.Tensor): 输入的多维 ID 张量，形状为 [..., n_axes]。

        返回:
            torch.Tensor: 嵌入后的张量，形状为 [..., n_axes, embed_dim]。
        """
        # 获取轴的数量
        n_axes = ids.shape[-1]
        # 对每个轴应用旋转位置编码（rope）进行嵌入，并将结果在倒数第三个维度上拼接
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,  # 在倒数第三个维度上拼接
        )
        # 在倒数第二个维度上增加一个维度，得到形状 [..., n_axes, 1, embed_dim]
        return emb.unsqueeze(2)


class PatchEmbed(nn.Module):
    """
    PatchEmbed 类用于将输入的潜在向量（latent）分割成小块（patch），并通过线性变换将其投影到更高的维度。

    参数:
        patch_size (int, optional): 每个小块的大小，默认为2。
        in_channels (int, optional): 输入通道数，默认为4。
        out_channels (int, optional): 输出通道数，默认为1024。
        dtype (torch.dtype, optional): 数据类型，默认为 None。
        device (torch.device, optional): 设备（CPU 或 GPU），默认为 None。
        operations (object, optional): 自定义操作对象，默认为 None。
    """
    def __init__(
        self,
        patch_size=2,
        in_channels=4,
        out_channels=1024,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        # 每个小块的大小
        self.patch_size = patch_size
        # 输出通道数
        self.out_channels = out_channels
        # 使用线性变换将输入通道数乘以小块大小的平方，映射到输出通道数
        self.proj = operations.Linear(in_channels * patch_size * patch_size, out_channels, bias=True, dtype=dtype, device=device)

    def forward(self, latent):
        """
        前向传播过程，将输入的潜在向量分割成小块并投影到更高的维度。

        参数:
            latent (torch.Tensor): 输入的潜在向量张量。

        返回:
            torch.Tensor: 投影后的张量。
        """
        # 通过线性变换进行投影
        latent = self.proj(latent)
        # 返回投影后的张量
        return latent


class PooledEmbed(nn.Module):
    """
    PooledEmbed 类用于对池化后的嵌入（pooled embedding）进行进一步的处理。
    它通过 TimestepEmbedding 对输入的池化嵌入进行时间步嵌入。

    参数:
        text_emb_dim (int): 文本嵌入的维度。
        hidden_size (int): 隐藏层的大小，用于时间步嵌入。
        dtype (torch.dtype, optional): 数据类型，默认为 None。
        device (torch.device, optional): 设备（CPU 或 GPU），默认为 None。
        operations (object, optional): 自定义操作对象，默认为 None。
    """
    def __init__(self, text_emb_dim, hidden_size, dtype=None, device=None, operations=None):
        super().__init__()
        # 初始化 TimestepEmbedding，用于将时间步嵌入到隐藏层大小
        self.pooled_embedder = TimestepEmbedding(in_channels=text_emb_dim, time_embed_dim=hidden_size, dtype=dtype, device=device, operations=operations)

    def forward(self, pooled_embed):
        """
        前向传播过程，对池化后的嵌入进行时间步嵌入。

        参数:
            pooled_embed (torch.Tensor): 池化后的嵌入张量。

        返回:
            torch.Tensor: 时间步嵌入后的张量。
        """
        # 返回时间步嵌入后的张量
        return self.pooled_embedder(pooled_embed)


class TimestepEmbed(nn.Module):
    """
    TimestepEmbed 类用于将时间步嵌入到高维空间中。
    它首先通过 Timesteps 对时间步进行投影，然后通过 TimestepEmbedding 进行进一步嵌入。

    参数:
        hidden_size (int): 隐藏层的大小。
        frequency_embedding_size (int, optional): 频率嵌入的维度，默认为256。
        dtype (torch.dtype, optional): 数据类型，默认为 None。
        device (torch.device, optional): 设备（CPU 或 GPU），默认为 None。
        operations (object, optional): 自定义操作对象，默认为 None。
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=None, device=None, operations=None):
        super().__init__()
        # 初始化 Timesteps，用于对时间步进行投影
        self.time_proj = Timesteps(num_channels=frequency_embedding_size, flip_sin_to_cos=True, downscale_freq_shift=0)
        # 初始化 TimestepEmbedding，用于将投影后的时间步嵌入到隐藏层大小
        self.timestep_embedder = TimestepEmbedding(in_channels=frequency_embedding_size, time_embed_dim=hidden_size, dtype=dtype, device=device, operations=operations)

    def forward(self, timesteps, wdtype):
        """
        前向传播过程，将时间步嵌入到高维空间中。

        参数:
            timesteps (torch.Tensor): 输入的时间步张量。
            wdtype (torch.dtype): 数据类型。

        返回:
            torch.Tensor: 时间步嵌入后的张量。
        """
        # 对时间步进行投影并转换数据类型
        t_emb = self.time_proj(timesteps).to(dtype=wdtype)
        # 对投影后的时间步进行嵌入
        t_emb = self.timestep_embedder(t_emb)
        # 返回嵌入后的时间步张量
        return t_emb


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    """
    注意力函数，使用优化后的注意力机制计算注意力输出。

    参数:
        query (torch.Tensor): 查询张量，形状为 (batch_size, ..., dim)。
        key (torch.Tensor): 键张量，形状为 (batch_size, ..., dim)。
        value (torch.Tensor): 值张量，形状为 (batch_size, ..., dim)。

    返回:
        torch.Tensor: 注意力输出，形状为 (batch_size, ..., dim)。
    """
    return optimized_attention(query.view(query.shape[0], -1, query.shape[-1] * query.shape[-2]), key.view(key.shape[0], -1, key.shape[-1] * key.shape[-2]), value.view(value.shape[0], -1, value.shape[-1] * value.shape[-2]), query.shape[2])


class HiDreamAttnProcessor_flashattn:
    """
    HiDreamAttnProcessor_flashattn 类是一个注意力处理器，通常用于处理类似 SD3 的自注意力投影。
    它实现了使用 FlashAttention 的高效注意力计算。

    参数:
        attn: 注意力模块，包含查询、键、值的线性变换和归一化参数。
        image_tokens (torch.FloatTensor): 图像标记张量，形状为 (batch_size, n_tokens, embed_dim)。
        image_tokens_masks (torch.FloatTensor, optional): 图像标记掩码张量，形状为 (batch_size, n_tokens)，默认为 None。
        text_tokens (torch.FloatTensor, optional): 文本标记张量，形状为 (batch_size, n_tokens, embed_dim)，默认为 None。
        rope (torch.FloatTensor, optional): 旋转位置编码张量，形状为 (batch_size, n_tokens, embed_dim)，默认为 None。
        *args: 其他位置参数。
        **kwargs: 其他关键字参数。
    """
    def __call__(
        self,
        attn,
        image_tokens: torch.FloatTensor,
        image_tokens_masks: Optional[torch.FloatTensor] = None,
        text_tokens: Optional[torch.FloatTensor] = None,
        rope: torch.FloatTensor = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        前向传播过程，计算注意力输出。

        参数:
            attn: 注意力模块。
            image_tokens (torch.FloatTensor): 图像标记张量。
            image_tokens_masks (torch.FloatTensor, optional): 图像标记掩码张量，默认为 None。
            text_tokens (torch.FloatTensor, optional): 文本标记张量，默认为 None。
            rope (torch.FloatTensor, optional): 旋转位置编码张量，默认为 None。
            *args: 其他位置参数。
            **kwargs: 其他关键字参数。

        返回:
            torch.FloatTensor: 注意力输出张量。
        """
        # 获取数据类型
        dtype = image_tokens.dtype
        # 获取批大小
        batch_size = image_tokens.shape[0]

        # 计算查询、键、值
        query_i = attn.q_rms_norm(attn.to_q(image_tokens)).to(dtype=dtype)  # 计算图像查询
        key_i = attn.k_rms_norm(attn.to_k(image_tokens)).to(dtype=dtype)  # 计算图像键
        value_i = attn.to_v(image_tokens)  # 计算图像值

        # 获取键的内部维度
        inner_dim = key_i.shape[-1]
        # 计算每个头的维度
        head_dim = inner_dim // attn.heads

        # 重塑查询、键、值以适应多头注意力机制
        query_i = query_i.view(batch_size, -1, attn.heads, head_dim)
        key_i = key_i.view(batch_size, -1, attn.heads, head_dim)
        value_i = value_i.view(batch_size, -1, attn.heads, head_dim)
        if image_tokens_masks is not None:
            # 应用图像标记掩码
            key_i = key_i * image_tokens_masks.view(batch_size, -1, 1, 1)

        if not attn.single:
            # 如果不是单一模式，则处理文本标记
            query_t = attn.q_rms_norm_t(attn.to_q_t(text_tokens)).to(dtype=dtype) # 计算文本查询
            key_t = attn.k_rms_norm_t(attn.to_k_t(text_tokens)).to(dtype=dtype) # 计算文本键
            value_t = attn.to_v_t(text_tokens) # 计算文本值

            query_t = query_t.view(batch_size, -1, attn.heads, head_dim)
            key_t = key_t.view(batch_size, -1, attn.heads, head_dim)
            value_t = value_t.view(batch_size, -1, attn.heads, head_dim)

            num_image_tokens = query_i.shape[1]
            num_text_tokens = query_t.shape[1]

            # 拼接图像和文本查询
            query = torch.cat([query_i, query_t], dim=1)
            # 拼接图像和文本键
            key = torch.cat([key_i, key_t], dim=1)
            # 拼接图像和文本值
            value = torch.cat([value_i, value_t], dim=1)

        else:

            # 仅使用图像查询
            query = query_i
            # 仅使用图像键
            key = key_i
            # 仅使用图像值
            value = value_i

        if query.shape[-1] == rope.shape[-3] * 2:
            # 应用旋转位置编码
            query, key = apply_rope(query, key, rope)
        else:
            query_1, query_2 = query.chunk(2, dim=-1)
            key_1, key_2 = key.chunk(2, dim=-1)
            # 应用旋转位置编码到第一部分
            query_1, key_1 = apply_rope(query_1, key_1, rope)
            # 拼接回查询张量
            query = torch.cat([query_1, query_2], dim=-1)
            # 拼接回键张量
            key = torch.cat([key_1, key_2], dim=-1)

        # 计算注意力输出
        hidden_states = attention(query, key, value)

        if not attn.single:
            # 分割图像和文本注意力输出
            hidden_states_i, hidden_states_t = torch.split(hidden_states, [num_image_tokens, num_text_tokens], dim=1)
            # 对图像注意力输出进行投影
            hidden_states_i = attn.to_out(hidden_states_i)
            # 对文本注意力输出进行投影
            hidden_states_t = attn.to_out_t(hidden_states_t)
            # 返回图像和文本注意力输出
            return hidden_states_i, hidden_states_t
        else:
            # 对注意力输出进行投影
            hidden_states = attn.to_out(hidden_states)
            # 返回注意力输出
            return hidden_states


class HiDreamAttention(nn.Module):
    """
    HiDreamAttention 类实现了多头自注意力机制，支持多种配置选项，如向上缩放注意力计算和 softmax 运算。

    参数:
        query_dim (int): 查询向量的维度。
        heads (int, optional): 多头注意力的头数，默认为8。
        dim_head (int, optional): 每个头的维度，默认为64。
        upcast_attention (bool, optional): 是否在计算注意力时进行向上类型转换，默认为 False。
        upcast_softmax (bool, optional): 是否在 softmax 运算时进行向上类型转换，默认为 False。
        scale_qk (bool, optional): 是否对查询和键进行缩放，默认为 True。
        eps (float, optional): RMSNorm 的小常数，防止除零错误，默认为 1e-5。
        processor (object, optional): 注意力处理器，默认为 None。
        out_dim (int, optional): 输出维度。如果为 None，则默认为 `query_dim`，默认为 None。
        single (bool, optional): 是否为单一模式（仅处理图像或文本），默认为 False。
        dtype (torch.dtype, optional): 数据类型，默认为 None。
        device (torch.device, optional): 设备（CPU 或 GPU），默认为 None。
        operations (object, optional): 自定义操作对象，默认为 None。
    """
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        scale_qk: bool = True,
        eps: float = 1e-5,
        processor = None,
        out_dim: int = None,
        single: bool = False,
        dtype=None, device=None, operations=None
    ):
        # super(Attention, self).__init__()
        super().__init__()
        # 计算内部维度
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        # 查询向量的维度
        self.query_dim = query_dim
        # 是否在计算注意力时进行向上类型转换
        self.upcast_attention = upcast_attention
        # 是否在 softmax 运算时进行向上类型转换
        self.upcast_softmax = upcast_softmax
        # 输出维度
        self.out_dim = out_dim if out_dim is not None else query_dim

        # 是否对查询和键进行缩放
        self.scale_qk = scale_qk
        # 缩放因子，如果 scale_qk 为 True，则为 dim_head 的 -0.5 次方，否则为1.0
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        # 多头注意力的头数
        self.heads = out_dim // dim_head if out_dim is not None else heads
        # 可切片的头维度
        self.sliceable_head_dim = heads
        # 是否为单一模式
        self.single = single

        # 获取线性层类
        linear_cls = operations.Linear
        # 存储线性层类
        self.linear_cls = linear_cls
        # 查询线性变换
        self.to_q = linear_cls(query_dim, self.inner_dim, dtype=dtype, device=device)
        # 键线性变换
        self.to_k = linear_cls(self.inner_dim, self.inner_dim, dtype=dtype, device=device)
        # 值线性变换
        self.to_v = linear_cls(self.inner_dim, self.inner_dim, dtype=dtype, device=device)
        # 输出线性变换
        self.to_out = linear_cls(self.inner_dim, self.out_dim, dtype=dtype, device=device)
        # 查询的 RMSNorm 归一化
        self.q_rms_norm = operations.RMSNorm(self.inner_dim, eps, dtype=dtype, device=device)
        # 键的 RMSNorm 归一化
        self.k_rms_norm = operations.RMSNorm(self.inner_dim, eps, dtype=dtype, device=device)

        if not single:
            # 如果不是单一模式，则初始化处理文本标记的线性变换和归一化层
            self.to_q_t = linear_cls(query_dim, self.inner_dim, dtype=dtype, device=device)
            self.to_k_t = linear_cls(self.inner_dim, self.inner_dim, dtype=dtype, device=device)
            self.to_v_t = linear_cls(self.inner_dim, self.inner_dim, dtype=dtype, device=device)
            self.to_out_t = linear_cls(self.inner_dim, self.out_dim, dtype=dtype, device=device)
            self.q_rms_norm_t = operations.RMSNorm(self.inner_dim, eps, dtype=dtype, device=device)
            self.k_rms_norm_t = operations.RMSNorm(self.inner_dim, eps, dtype=dtype, device=device)

        # 注意力处理器
        self.processor = processor

    def forward(
        self,
        norm_image_tokens: torch.FloatTensor,
        image_tokens_masks: torch.FloatTensor = None,
        norm_text_tokens: torch.FloatTensor = None,
        rope: torch.FloatTensor = None,
    ) -> torch.Tensor:
        """
        前向传播过程，计算多头自注意力。

        参数:
            norm_image_tokens (torch.FloatTensor): 归一化后的图像标记张量。
            image_tokens_masks (torch.FloatTensor, optional): 图像标记掩码张量，默认为 None。
            norm_text_tokens (torch.FloatTensor, optional): 归一化后的文本标记张量，默认为 None。
            rope (torch.FloatTensor, optional): 旋转位置编码张量，默认为 None。

        返回:
            torch.Tensor: 注意力输出张量。
        """
        # 使用处理器处理注意力
        return self.processor(
            self,
            image_tokens = norm_image_tokens,
            image_tokens_masks = image_tokens_masks,
            text_tokens = norm_text_tokens,
            rope = rope,
        )


class FeedForwardSwiGLU(nn.Module):
    """
    FeedForwardSwiGLU 类实现了带有 Swish-GLU 激活函数的前馈神经网络（Feed-Forward Network, FFN）。

    参数:
        dim (int): 输入和输出的维度。
        hidden_dim (int): 隐藏层的维度。
        multiple_of (int, optional): 隐藏层维度的倍数，默认为256。
        ffn_dim_multiplier (float, optional): FFN 维度乘数，默认为 None。
        dtype (torch.dtype, optional): 数据类型，默认为 None。
        device (torch.device, optional): 设备（CPU 或 GPU），默认为 None。
        operations (object, optional): 自定义操作对象，默认为 None。
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        # 计算隐藏层维度
        hidden_dim = int(2 * hidden_dim / 3)
        # 自定义维度因子乘数
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * (
            (hidden_dim + multiple_of - 1) // multiple_of  # 调整隐藏层维度为 multiple_of 的倍数
        )

        # 第一个线性层
        self.w1 = operations.Linear(dim, hidden_dim, bias=False, dtype=dtype, device=device)
        # 第二个线性层
        self.w2 = operations.Linear(hidden_dim, dim, bias=False, dtype=dtype, device=device)
        # 第三个线性层
        self.w3 = operations.Linear(dim, hidden_dim, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        """
        前向传播过程。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: FFN 处理后的输出张量。
        """
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


# Modified from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
class MoEGate(nn.Module):
    """
    MoEGate 类实现了专家混合门控机制，用于路由输入到不同的专家（Experts）。

    参数:
        embed_dim (int): 输入嵌入的维度。
        num_routed_experts (int, optional): 路由的专家数量，默认为4。
        num_activated_experts (int, optional): 激活的专家数量，默认为2。
        aux_loss_alpha (float, optional): 辅助损失的权重，默认为0.01。
        dtype (torch.dtype, optional): 数据类型，默认为 None。
        device (torch.device, optional): 设备（CPU 或 GPU），默认为 None。
        operations (object, optional): 自定义操作对象，默认为 None。
    """
    def __init__(self, embed_dim, num_routed_experts=4, num_activated_experts=2, aux_loss_alpha=0.01, dtype=None, device=None, operations=None):
        super().__init__()
        # 激活的专家数量
        self.top_k = num_activated_experts
        # 路由的专家数量
        self.n_routed_experts = num_routed_experts

        # 评分函数类型
        self.scoring_func = 'softmax'
        # 辅助损失的权重
        self.alpha = aux_loss_alpha
        # 是否使用序列辅助损失
        self.seq_aux = False

        # topk 选择算法
        # 是否对 topk 概率进行归一化
        self.norm_topk_prob = False
        # 门控维度
        self.gating_dim = embed_dim
        # 权重参数
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim), dtype=dtype, device=device))
        # 重置参数
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass
        # import torch.nn.init  as init
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # 使用 kaiming 初始化权重

    def forward(self, hidden_states):
        """
        前向传播过程。

        参数:
            hidden_states (torch.Tensor): 输入隐藏状态，形状为 (batch_size, seq_len, embed_dim)。

        返回:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                - topk_idx (torch.Tensor): 激活专家的索引。
                - topk_weight (torch.Tensor): 激活专家的权重。
                - aux_loss (Optional[torch.Tensor]): 辅助损失，默认为 None。
        """
        # 获取批量大小、序列长度和嵌入维度
        bsz, seq_len, h = hidden_states.shape

        ### 计算门控得分
        hidden_states = hidden_states.view(-1, h) # 重塑为 (batch_size * seq_len, embed_dim)
        # 计算线性变换后的得分
        logits = F.linear(hidden_states, model_management.cast_to(self.weight, dtype=hidden_states.dtype, device=hidden_states.device), None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1) # 使用 softmax 计算概率
        else:
            # 如果不支持的评分函数，则抛出异常
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        ### 选择 top-k 专家
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        ### 归一化门控概率
        if self.top_k > 1 and self.norm_topk_prob:
            # 计算分母，防止除零
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            # 归一化
            topk_weight = topk_weight / denominator

        # 辅助损失，默认为 None
        aux_loss = None
        # 返回激活专家的索引、权重和辅助损失
        return topk_idx, topk_weight, aux_loss


# Modified from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
class MOEFeedForwardSwiGLU(nn.Module):
    """
    MOEFeedForwardSwiGLU 类实现了基于专家混合（Mixture of Experts, MoE）的带有 Swish-GLU 激活函数的前馈神经网络。
    该类结合了共享专家和路由专家，通过门控机制将输入路由到不同的专家进行处理。

    参数:
        dim (int): 输入和输出的维度。
        hidden_dim (int): 隐藏层的维度。
        num_routed_experts (int): 路由的专家数量。
        num_activated_experts (int): 激活的专家数量。
        dtype (torch.dtype, optional): 数据类型，默认为 None。
        device (torch.device, optional): 设备（CPU 或 GPU），默认为 None。
        operations (object, optional): 自定义操作对象，默认为 None。
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_routed_experts: int,
        num_activated_experts: int,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        # 初始化共享专家，隐藏层维度为 hidden_dim // 2
        self.shared_experts = FeedForwardSwiGLU(dim, hidden_dim // 2, dtype=dtype, device=device, operations=operations)
        # 初始化路由专家列表，每个专家的隐藏层维度为 hidden_dim
        self.experts = nn.ModuleList([FeedForwardSwiGLU(dim, hidden_dim, dtype=dtype, device=device, operations=operations) for i in range(num_routed_experts)])
        # 初始化门控机制，用于路由输入到不同的专家
        self.gate = MoEGate(
            embed_dim = dim,
            num_routed_experts = num_routed_experts,
            num_activated_experts = num_activated_experts,
            dtype=dtype, device=device, operations=operations
        )
        # 激活的专家数量
        self.num_activated_experts = num_activated_experts

    def forward(self, x):
        """
        前向传播过程，通过门控机制将输入路由到不同的专家进行处理。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 处理后的输出张量。
        """
        # 获取数据类型
        wtype = x.dtype
        # 保留原始输入，用于残差连接
        identity = x
        # 记录原始形状
        orig_shape = x.shape
        # 通过门控机制获取激活专家的索引、权重和辅助损失
        topk_idx, topk_weight, aux_loss = self.gate(x)
        # 重塑输入张量为 (batch_size * seq_len, dim)
        x = x.view(-1, x.shape[-1])
        # 展平激活专家索引
        flat_topk_idx = topk_idx.view(-1)

        if True:  # self.training: # TODO: 检查哪个分支执行更快
            # 如果在训练模式下，使用重复插值将输入复制以匹配激活专家的数量
            x = x.repeat_interleave(self.num_activated_experts, dim=0)
            # 创建一个与 x 形状相同的空张量
            y = torch.empty_like(x, dtype=wtype)

            for i, expert in enumerate(self.experts):
                # 对应专家处理输入，并转换为原始数据类型
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(dtype=wtype)

            # 对专家输出进行加权求和
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            # 重塑回原始形状并转换数据类型
            y =  y.view(*orig_shape).to(dtype=wtype)
            #y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            # 如果在推理模式下，使用不同的处理方式
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        # 将共享专家的输出与路由专家的输出相加，实现残差连接
        y = y + self.shared_experts(identity)
        return y
    
    # 禁用梯度计算，节省内存并加快推理速度
    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        推理过程中处理专家混合的辅助方法。

        参数:
            x (torch.Tensor): 输入张量。
            flat_expert_indices (torch.Tensor): 展平后的激活专家索引。
            flat_expert_weights (torch.Tensor): 激活专家的权重。

        返回:
            torch.Tensor: 处理后的输出张量。
        """
        # 初始化专家缓存
        expert_cache = torch.zeros_like(x)
        # 对专家索引进行排序
        idxs = flat_expert_indices.argsort()
        # 计算每个专家处理的 token 数量
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 计算每个 token 对应的专家索引
        token_idxs = idxs // self.num_activated_experts
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                # 如果没有 token 被分配给当前专家，则跳过
                continue
            # 获取当前专家
            expert = self.experts[i]
            # 获取分配给当前专家的 token 索引
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 获取分配给当前专家的 token
            expert_tokens = x[exp_token_idx]
            # 处理 token
            expert_out = expert(expert_tokens)
            # 对输出进行加权
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])

            # 处理不同的数据类型（如 fp16）
            expert_cache = expert_cache.to(expert_out.dtype)
            # 将专家输出累加到缓存中
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
        
        # 返回处理后的缓存
        return expert_cache


class TextProjection(nn.Module):
    """
    TextProjection 类用于将文本特征投影到隐藏层空间。

    参数:
        in_features (int): 输入特征的维度。
        hidden_size (int): 隐藏层的维度。
        dtype (torch.dtype, optional): 数据类型，默认为 None。
        device (torch.device, optional): 设备（CPU 或 GPU），默认为 None。
        operations (object, optional): 自定义操作对象，默认为 None。
    """
    def __init__(self, in_features, hidden_size, dtype=None, device=None, operations=None):
        super().__init__()
        # 初始化线性层，将输入特征投影到隐藏层空间
        self.linear = operations.Linear(in_features=in_features, out_features=hidden_size, bias=False, dtype=dtype, device=device)

    def forward(self, caption):
        """
        前向传播过程，将文本特征投影到隐藏层空间。

        参数:
            caption (torch.Tensor): 输入的文本特征。

        返回:
            torch.Tensor: 投影后的隐藏状态。
        """
        # 应用线性变换
        hidden_states = self.linear(caption)
        # 返回投影后的隐藏状态
        return hidden_states


class BlockType:
    """
    BlockType 类定义了 Transformer 块的类型。
    """
    TransformerBlock = 1  # 标准 Transformer 块
    SingleTransformerBlock = 2  # 单个 Transformer 块


class HiDreamImageSingleTransformerBlock(nn.Module):
    """
    HiDreamImageSingleTransformerBlock 类实现了单个 Transformer 块，用于处理图像标记。
    该块结合了自适应层归一化（adaLN）、多头自注意力机制和前馈神经网络（FFN）。

    参数:
        dim (int): 输入和输出的维度。
        num_attention_heads (int): 多头注意力的头数。
        attention_head_dim (int): 每个注意力头的维度。
        num_routed_experts (int, optional): 路由的专家数量，默认为4。
        num_activated_experts (int, optional): 激活的专家数量，默认为2。
        dtype (torch.dtype, optional): 数据类型，默认为 None。
        device (torch.device, optional): 设备（CPU 或 GPU），默认为 None。
        operations (object, optional): 自定义操作对象，默认为 None。
    """
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        # 多头注意力的头数
        self.num_attention_heads = num_attention_heads
        # 自适应层归一化（adaLN）调制模块
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            operations.Linear(dim, 6 * dim, bias=True, dtype=dtype, device=device)
        )

        # 1. Attention
        # 层归一化
        self.norm1_i = operations.LayerNorm(dim, eps = 1e-06, elementwise_affine = False, dtype=dtype, device=device)
        # 多头自注意力机制
        self.attn1 = HiDreamAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            processor = HiDreamAttnProcessor_flashattn(),
            single = True,
            dtype=dtype, device=device, operations=operations
        )

        # 3. Feed-forward
        # 层归一化
        self.norm3_i = operations.LayerNorm(dim, eps = 1e-06, elementwise_affine = False, dtype=dtype, device=device)
        # 基于专家混合的前馈神经网络
        if num_routed_experts > 0:
            self.ff_i = MOEFeedForwardSwiGLU(
                dim = dim,
                hidden_dim = 4 * dim,
                num_routed_experts = num_routed_experts,
                num_activated_experts = num_activated_experts,
                dtype=dtype, device=device, operations=operations
            )
        else:
            # 标准前馈神经网络
            self.ff_i = FeedForwardSwiGLU(dim = dim, hidden_dim = 4 * dim, dtype=dtype, device=device, operations=operations)

    def forward(
        self,
        image_tokens: torch.FloatTensor,
        image_tokens_masks: Optional[torch.FloatTensor] = None,
        text_tokens: Optional[torch.FloatTensor] = None,
        adaln_input: Optional[torch.FloatTensor] = None,
        rope: torch.FloatTensor = None,

    ) -> torch.FloatTensor:
        """
        前向传播过程，处理图像标记。

        参数:
            image_tokens (torch.FloatTensor): 输入的图像标记。
            image_tokens_masks (torch.FloatTensor, optional): 图像标记掩码，默认为 None。
            text_tokens (torch.FloatTensor, optional): 输入的文本标记，默认为 None。
            adaln_input (torch.FloatTensor, optional): 自适应层归一化输入，默认为 None。
            rope (torch.FloatTensor, optional): 旋转位置编码，默认为 None。

        返回:
            torch.FloatTensor: 处理后的图像标记。
        """
        # 获取数据类型
        wtype = image_tokens.dtype
        # 获取自适应层归一化调制参数
        shift_msa_i, scale_msa_i, gate_msa_i, shift_mlp_i, scale_mlp_i, gate_mlp_i = \
            self.adaLN_modulation(adaln_input)[:,None].chunk(6, dim=-1)

        # 1. MM-Attention
        # 层归一化
        norm_image_tokens = self.norm1_i(image_tokens).to(dtype=wtype)
        # 自适应调整
        norm_image_tokens = norm_image_tokens * (1 + scale_msa_i) + shift_msa_i
        # 应用多头自注意力
        attn_output_i = self.attn1(
            norm_image_tokens,
            image_tokens_masks,
            rope = rope,
        )
        # 残差连接
        image_tokens = gate_msa_i * attn_output_i + image_tokens

        # 2. Feed-forward
        # 层归一化
        norm_image_tokens = self.norm3_i(image_tokens).to(dtype=wtype)
        # 自适应调整
        norm_image_tokens = norm_image_tokens * (1 + scale_mlp_i) + shift_mlp_i
        # 应用前馈神经网络
        ff_output_i = gate_mlp_i * self.ff_i(norm_image_tokens.to(dtype=wtype))
        # 残差连接
        image_tokens = ff_output_i + image_tokens
        # 返回处理后的图像标记
        return image_tokens


class HiDreamImageTransformerBlock(nn.Module):
    """
    HiDreamImageTransformerBlock 类实现了用于处理图像和文本标记的 Transformer 块。
    该块结合了自适应层归一化（adaLN）、多头自注意力机制和前馈神经网络（FFN），并支持图像和文本的交叉注意力。

    参数:
        dim (int): 输入和输出的维度。
        num_attention_heads (int): 多头注意力的头数。
        attention_head_dim (int): 每个注意力头的维度。
        num_routed_experts (int, optional): 路由的专家数量，默认为4。
        num_activated_experts (int, optional): 激活的专家数量，默认为2。
        dtype (torch.dtype, optional): 数据类型，默认为 None。
        device (torch.device, optional): 设备（CPU 或 GPU），默认为 None。
        operations (object, optional): 自定义操作对象，默认为 None。
    """
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        # 多头注意力的头数
        self.num_attention_heads = num_attention_heads
        # 自适应层归一化（adaLN）调制模块，包含 SiLU 激活函数和线性变换
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            # 线性层，输出维度为 12 * dim
            operations.Linear(dim, 12 * dim, bias=True, dtype=dtype, device=device)
        )
        # 初始化权重和偏置为零（可选）
        # nn.init.zeros_(self.adaLN_modulation[1].weight)
        # nn.init.zeros_(self.adaLN_modulation[1].bias)

        # 1. Attention
        # 图像标记的层归一化
        self.norm1_i = operations.LayerNorm(dim, eps = 1e-06, elementwise_affine = False, dtype=dtype, device=device)
        # 文本标记的层归一化
        self.norm1_t = operations.LayerNorm(dim, eps = 1e-06, elementwise_affine = False, dtype=dtype, device=device)
        # 多头自注意力机制
        self.attn1 = HiDreamAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            processor = HiDreamAttnProcessor_flashattn(),  # 使用 FlashAttention 的处理器
            single = False,  # 不是单一模式，处理图像和文本
            dtype=dtype, device=device, operations=operations
        )

        # 3. Feed-forward
        # 图像标记的层归一化
        self.norm3_i = operations.LayerNorm(dim, eps = 1e-06, elementwise_affine = False, dtype=dtype, device=device)
        # 基于专家混合的前馈神经网络
        if num_routed_experts > 0:
            self.ff_i = MOEFeedForwardSwiGLU(
                dim = dim,
                hidden_dim = 4 * dim,
                num_routed_experts = num_routed_experts,
                num_activated_experts = num_activated_experts,
                dtype=dtype, device=device, operations=operations
            )
        else:
            # 标准前馈神经网络
            self.ff_i = FeedForwardSwiGLU(dim = dim, hidden_dim = 4 * dim, dtype=dtype, device=device, operations=operations)
        # 文本标记的层归一化
        self.norm3_t = operations.LayerNorm(dim, eps = 1e-06, elementwise_affine = False)
        # 文本标记的前馈神经网络
        self.ff_t = FeedForwardSwiGLU(dim = dim, hidden_dim = 4 * dim, dtype=dtype, device=device, operations=operations)

    def forward(
        self,
        image_tokens: torch.FloatTensor,
        image_tokens_masks: Optional[torch.FloatTensor] = None,
        text_tokens: Optional[torch.FloatTensor] = None,
        adaln_input: Optional[torch.FloatTensor] = None,
        rope: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        前向传播过程，处理图像和文本标记。

        参数:
            image_tokens (torch.FloatTensor): 输入的图像标记。
            image_tokens_masks (torch.FloatTensor, optional): 图像标记掩码，默认为 None。
            text_tokens (torch.FloatTensor, optional): 输入的文本标记，默认为 None。
            adaln_input (torch.FloatTensor, optional): 自适应层归一化输入，默认为 None。
            rope (torch.FloatTensor, optional): 旋转位置编码，默认为 None。

        返回:
            Tuple[torch.FloatTensor, torch.FloatTensor]: 处理后的图像和文本标记。
        """
        # 获取数据类型
        wtype = image_tokens.dtype
        # 获取自适应层归一化调制参数，并分割为12个部分
        shift_msa_i, scale_msa_i, gate_msa_i, shift_mlp_i, scale_mlp_i, gate_mlp_i, \
        shift_msa_t, scale_msa_t, gate_msa_t, shift_mlp_t, scale_mlp_t, gate_mlp_t = \
            self.adaLN_modulation(adaln_input)[:,None].chunk(12, dim=-1)

        # 1. MM-Attention
        # 层归一化
        norm_image_tokens = self.norm1_i(image_tokens).to(dtype=wtype)
        # 自适应调整
        norm_image_tokens = norm_image_tokens * (1 + scale_msa_i) + shift_msa_i
        # 层归一化
        norm_text_tokens = self.norm1_t(text_tokens).to(dtype=wtype)
        # 自适应调整
        norm_text_tokens = norm_text_tokens * (1 + scale_msa_t) + shift_msa_t

        # 应用多头自注意力
        attn_output_i, attn_output_t = self.attn1(
            norm_image_tokens,
            image_tokens_masks,
            norm_text_tokens,
            rope = rope,
        )

        # 残差连接
        image_tokens = gate_msa_i * attn_output_i + image_tokens
        text_tokens = gate_msa_t * attn_output_t + text_tokens

        # 2. Feed-forward
        # 层归一化
        norm_image_tokens = self.norm3_i(image_tokens).to(dtype=wtype)
        # 自适应调整
        norm_image_tokens = norm_image_tokens * (1 + scale_mlp_i) + shift_mlp_i
        # 层归一化
        norm_text_tokens = self.norm3_t(text_tokens).to(dtype=wtype)
        # 自适应调整
        norm_text_tokens = norm_text_tokens * (1 + scale_mlp_t) + shift_mlp_t

        # 应用前馈神经网络
        ff_output_i = gate_mlp_i * self.ff_i(norm_image_tokens)
        ff_output_t = gate_mlp_t * self.ff_t(norm_text_tokens)

        # 残差连接
        image_tokens = ff_output_i + image_tokens
        text_tokens = ff_output_t + text_tokens

        # 返回处理后的图像和文本标记
        return image_tokens, text_tokens


class HiDreamImageBlock(nn.Module):
    """
    HiDreamImageBlock 类是一个封装类，用于根据指定的块类型创建不同的 Transformer 块。

    参数:
        dim (int): 输入和输出的维度。
        num_attention_heads (int): 多头注意力的头数。
        attention_head_dim (int): 每个注意力头的维度。
        num_routed_experts (int, optional): 路由的专家数量，默认为4。
        num_activated_experts (int, optional): 激活的专家数量，默认为2。
        block_type (BlockType, optional): 块类型，默认为 TransformerBlock。
        dtype (torch.dtype, optional): 数据类型，默认为 None。
        device (torch.device, optional): 设备（CPU 或 GPU），默认为 None。
        operations (object, optional): 自定义操作对象，默认为 None。
    """
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
        block_type: BlockType = BlockType.TransformerBlock,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        block_classes = {
            BlockType.TransformerBlock: HiDreamImageTransformerBlock, # 标准 Transformer 块
            BlockType.SingleTransformerBlock: HiDreamImageSingleTransformerBlock, # 单个 Transformer 块
        }

        # 根据 block_type 创建相应的 Transformer 块
        self.block = block_classes[block_type](
            dim,
            num_attention_heads,
            attention_head_dim,
            num_routed_experts,
            num_activated_experts,
            dtype=dtype, device=device, operations=operations
        )

    def forward(
        self,
        image_tokens: torch.FloatTensor,
        image_tokens_masks: Optional[torch.FloatTensor] = None,
        text_tokens: Optional[torch.FloatTensor] = None,
        adaln_input: torch.FloatTensor = None,
        rope: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        前向传播过程，根据指定的块类型处理输入数据。

        参数:
            image_tokens (torch.FloatTensor): 输入的图像标记。
            image_tokens_masks (torch.FloatTensor, optional): 图像标记掩码，默认为 None。
            text_tokens (torch.FloatTensor, optional): 输入的文本标记，默认为 None。
            adaln_input (torch.FloatTensor): 自适应层归一化输入。
            rope (torch.FloatTensor, optional): 旋转位置编码，默认为 None。

        返回:
            torch.FloatTensor: 处理后的输出数据。
        """
        # 调用相应的 Transformer 块进行处理
        return self.block(
            image_tokens,
            image_tokens_masks,
            text_tokens,
            adaln_input,
            rope,
        )


class HiDreamImageTransformer2DModel(nn.Module):
    """
    HiDreamImageTransformer2DModel 类是一个基于 Transformer 的2D图像模型，用于处理图像和文本输入。
    该模型结合了时间步嵌入、文本嵌入、图像小块嵌入、旋转位置编码（RoPE）以及多头自注意力机制。

    参数:
        patch_size (int, optional): 图像小块的大小，默认为 None。
        in_channels (int, optional): 输入图像的通道数，默认为64。
        out_channels (int, optional): 输出图像的通道数。如果为 None，则默认为 `in_channels`，默认为 None。
        num_layers (int, optional): Transformer 层的数量，默认为16。
        num_single_layers (int, optional): 单个 Transformer 层的数量，默认为32。
        attention_head_dim (int, optional): 每个注意力头的维度，默认为128。
        num_attention_heads (int, optional): 多头注意力的头数，默认为20。
        caption_channels (List[int], optional): 文本特征的通道数列表，默认为 None。
        text_emb_dim (int, optional): 文本嵌入的维度，默认为2048。
        num_routed_experts (int, optional): 路由的专家数量，默认为4。
        num_activated_experts (int, optional): 激活的专家数量，默认为2。
        axes_dims_rope (Tuple[int, int], optional): RoPE 的轴维度，默认为 (32, 32)。
        max_resolution (Tuple[int, int], optional): 最大图像分辨率，默认为 (128, 128)。
        llama_layers (List[int], optional): LLaMA 层的列表，默认为 None。
        image_model (object, optional): 图像模型，默认为 None。
        dtype (torch.dtype, optional): 数据类型，默认为 None。
        device (torch.device, optional): 设备（CPU 或 GPU），默认为 None。
        operations (object, optional): 自定义操作对象，默认为 None。
    """
    def __init__(
        self,
        patch_size: Optional[int] = None,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 16,
        num_single_layers: int = 32,
        attention_head_dim: int = 128,
        num_attention_heads: int = 20,
        caption_channels: List[int] = None,
        text_emb_dim: int = 2048,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
        axes_dims_rope: Tuple[int, int] = (32, 32),
        max_resolution: Tuple[int, int] = (128, 128),
        llama_layers: List[int] = None,
        image_model=None,
        dtype=None, device=None, operations=None
    ):
        # 图像小块的大小
        self.patch_size = patch_size
        # 多头注意力的头数
        self.num_attention_heads = num_attention_heads
        # 每个注意力头的维度
        self.attention_head_dim = attention_head_dim
        # Transformer 层的数量
        self.num_layers = num_layers
        # 单个 Transformer 层的数量
        self.num_single_layers = num_single_layers

        # 是否启用梯度检查点，默认为 False
        self.gradient_checkpointing = False

        super().__init__()
        # 数据类型
        self.dtype = dtype
        # 输出图像的通道数，如果未指定，则默认为 `in_channels`
        self.out_channels = out_channels or in_channels
        # 计算内部维度
        self.inner_dim = self.num_attention_heads * self.attention_head_dim
        # LLaMA 层的列表
        self.llama_layers = llama_layers

        # 时间步嵌入器
        self.t_embedder = TimestepEmbed(self.inner_dim, dtype=dtype, device=device, operations=operations)
        # 文本嵌入器
        self.p_embedder = PooledEmbed(text_emb_dim, self.inner_dim, dtype=dtype, device=device, operations=operations)
        # 图像小块嵌入器
        self.x_embedder = PatchEmbed(
            patch_size = patch_size,
            in_channels = in_channels,
            out_channels = self.inner_dim,
            dtype=dtype, device=device, operations=operations
        )
        # 旋转位置编码（RoPE）嵌入器
        self.pe_embedder = EmbedND(theta=10000, axes_dim=axes_dims_rope)

        # 双流 Transformer 块列表
        self.double_stream_blocks = nn.ModuleList(
            [
                HiDreamImageBlock(
                    dim = self.inner_dim,
                    num_attention_heads = self.num_attention_heads,
                    attention_head_dim = self.attention_head_dim,
                    num_routed_experts = num_routed_experts,
                    num_activated_experts = num_activated_experts,
                    block_type = BlockType.TransformerBlock,
                    dtype=dtype, device=device, operations=operations
                )
                for i in range(self.num_layers)
            ]
        )

        # 单流 Transformer 块列表
        self.single_stream_blocks = nn.ModuleList(
            [
                HiDreamImageBlock(
                    dim = self.inner_dim,
                    num_attention_heads = self.num_attention_heads,
                    attention_head_dim = self.attention_head_dim,
                    num_routed_experts = num_routed_experts,
                    num_activated_experts = num_activated_experts,
                    block_type = BlockType.SingleTransformerBlock,
                    dtype=dtype, device=device, operations=operations
                )
                for i in range(self.num_single_layers)
            ]
        )

        # 最后一层，用于输出投影
        self.final_layer = LastLayer(self.inner_dim, patch_size, self.out_channels, dtype=dtype, device=device, operations=operations)

        # 文本投影列表，根据 caption_channels 初始化
        caption_channels = [caption_channels[1], ] * (num_layers + num_single_layers) + [caption_channels[0], ]
        caption_projection = []
        for caption_channel in caption_channels:
            caption_projection.append(TextProjection(in_features=caption_channel, hidden_size=self.inner_dim, dtype=dtype, device=device, operations=operations))
        self.caption_projection = nn.ModuleList(caption_projection)
        # 计算最大序列长度
        self.max_seq = max_resolution[0] * max_resolution[1] // (patch_size * patch_size)

    def expand_timesteps(self, timesteps, batch_size, device):
        """
        扩展时间步张量，使其与批次大小匹配。

        参数:
            timesteps (Union[float, torch.Tensor]): 输入的时间步，可以是浮点数或张量。
            batch_size (int): 批次大小。
            device (torch.device): 设备。

        返回:
            torch.Tensor: 扩展后的时间步张量。
        """
        if not torch.is_tensor(timesteps):
            # 判断是否为 MPS 设备
            is_mps = device.type == "mps"
            if isinstance(timesteps, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            # 将时间步转换为张量
            timesteps = torch.tensor([timesteps], dtype=dtype, device=device)
        elif len(timesteps.shape) == 0:
            # 如果是标量张量，则增加一个维度
            timesteps = timesteps[None].to(device)
        # 以与 ONNX/Core ML 兼容的方式扩展到批次维度
        timesteps = timesteps.expand(batch_size)

        # 返回扩展后的时间步张量
        return timesteps

    def unpatchify(self, x: torch.Tensor, img_sizes: List[Tuple[int, int]]) -> List[torch.Tensor]:
        """
        将小块图像张量重新排列为原始图像张量。

        参数:
            x (torch.Tensor): 输入的小块图像张量，形状为 (batch_size, total_patches, patch_size*patch_size*channels)。
            img_sizes (List[Tuple[int, int]]): 每个图像的尺寸列表，元素为 (高度, 宽度)。

        返回:
            List[torch.Tensor]: 重新排列后的图像张量列表。
        """
        # 初始化列表，用于存储重新排列后的图像张量
        x_arr = []
        for i, img_size in enumerate(img_sizes):
            # 获取图像的高度和宽度
            pH, pW = img_size
            # 使用 einops 重新排列张量，将小块图像拼接到原始图像中
            x_arr.append(
                einops.rearrange(x[i, :pH*pW].reshape(1, pH, pW, -1), 'B H W (p1 p2 C) -> B C (H p1) (W p2)',
                    p1=self.patch_size, p2=self.patch_size)
            )
        # 将所有图像张量拼接在一起
        x = torch.cat(x_arr, dim=0)
        # 返回重新排列后的图像张量列表
        return x

    def patchify(self, x, max_seq, img_sizes=None):
        """
        将输入图像张量分割成小块，并生成相应的掩码。

        参数:
            x (torch.Tensor): 输入图像张量，形状为 (batch_size, channels, height, width)。
            max_seq (int): 最大序列长度。
            img_sizes (List[Tuple[int, int]], optional): 每个图像的尺寸列表，默认为 None。

        返回:
            Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
                - x (torch.Tensor): 分割后的小块图像张量。
                - x_masks (torch.Tensor): 小块图像掩码张量。
                - img_sizes (List[Tuple[int, int]]): 每个图像的尺寸列表。
        """
        # 计算小块面积的平方
        pz2 = self.patch_size * self.patch_size
        if isinstance(x, torch.Tensor):
            # 获取批量大小
            B = x.shape[0]
            device = x.device
            # 获取数据类型
            dtype = x.dtype
        else:
            # 获取批量大小
            B = len(x)
            device = x[0].device
            # 获取数据类型
            dtype = x[0].dtype
        # 初始化掩码张量
        x_masks = torch.zeros((B, max_seq), dtype=dtype, device=device)

        if img_sizes is not None:
            for i, img_size in enumerate(img_sizes):
                # 根据图像尺寸设置掩码
                x_masks[i, 0:img_size[0] * img_size[1]] = 1
            # 分割图像为小块
            x = einops.rearrange(x, 'B C S p -> B S (p C)', p=pz2)
        elif isinstance(x, torch.Tensor):
            # 计算图像的高度和宽度
            pH, pW = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
            # 分割图像为小块
            x = einops.rearrange(x, 'B C (H p1) (W p2) -> B (H W) (p1 p2 C)', p1=self.patch_size, p2=self.patch_size)
            # 生成图像尺寸列表
            img_sizes = [[pH, pW]] * B
            # 不需要掩码
            x_masks = None
        else:
            raise NotImplementedError
        
        # 返回分割后的小块图像、掩码和图像尺寸
        return x, x_masks, img_sizes

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        encoder_hidden_states_llama3=None,
        image_cond=None,
        control = None,
        transformer_options = {},
    ) -> torch.Tensor:
        """
        前向传播过程，处理输入图像和文本数据。

        参数:
            x (torch.Tensor): 输入图像张量，形状为 (batch_size, channels, height, width)。
            t (torch.Tensor): 时间步张量。
            y (torch.Tensor, optional): 池化嵌入张量，默认为 None。
            context (torch.Tensor, optional): 上下文张量，默认为 None。
            encoder_hidden_states_llama3 (torch.Tensor, optional): LLaMA3 编码器隐藏状态，默认为 None。
            image_cond (torch.Tensor, optional): 图像条件张量，默认为 None。
            control (torch.Tensor, optional): 控制张量，默认为 None。
            transformer_options (Dict, optional): Transformer 选项，默认为 {}。

        返回:
            torch.Tensor: 处理后的输出张量。
        """
        # 获取输入图像的批量大小、通道数、高度和宽度
        bs, c, h, w = x.shape
        if image_cond is not None:
            # 如果存在图像条件，则将其与输入图像拼接
            x = torch.cat([x, image_cond], dim=-1)
        # 对输入图像进行填充，以适应小块大小
        hidden_states = common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
        # 获取时间步
        timesteps = t
        # 获取池化嵌入
        pooled_embeds = y
        # 获取 T5 编码器隐藏状态
        T5_encoder_hidden_states = context

        # 初始化图像尺寸列表
        img_sizes = None

        # 空间前向传播
        batch_size = hidden_states.shape[0]  # 获取批量大小
        hidden_states_type = hidden_states.dtype  # 获取数据类型

        # 0. 时间步处理
        # 扩展时间步张量
        timesteps = self.expand_timesteps(timesteps, batch_size, hidden_states.device)
        # 通过时间步嵌入器进行嵌入
        timesteps = self.t_embedder(timesteps, hidden_states_type)
        # 通过文本嵌入器进行嵌入
        p_embedder = self.p_embedder(pooled_embeds)
        # 计算自适应层归一化输入
        adaln_input = timesteps + p_embedder

        # 将图像分割为小块并生成掩码
        hidden_states, image_tokens_masks, img_sizes = self.patchify(hidden_states, self.max_seq, img_sizes)
        if image_tokens_masks is None:
            # 获取图像的高度和宽度
            pH, pW = img_sizes[0]
            # 初始化图像 ID 张量
            img_ids = torch.zeros(pH, pW, 3, device=hidden_states.device)

            # 设置图像 ID
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(pH, device=hidden_states.device)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(pW, device=hidden_states.device)[None, :]

            # 扩展图像 ID 以匹配批量大小
            img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch_size)
        
        # 通过图像小块嵌入器进行嵌入
        hidden_states = self.x_embedder(hidden_states)

        # 重塑 LLaMA3 编码器隐藏状态
        encoder_hidden_states = encoder_hidden_states_llama3.movedim(1, 0)
        # 选择特定的 LLaMA3 编码器层
        encoder_hidden_states = [encoder_hidden_states[k] for k in self.llama_layers]

        if self.caption_projection is not None:
            # 初始化新的编码器隐藏状态列表
            new_encoder_hidden_states = []
            for i, enc_hidden_state in enumerate(encoder_hidden_states):
                # 通过文本投影层进行投影
                enc_hidden_state = self.caption_projection[i](enc_hidden_state)
                # 重塑张量
                enc_hidden_state = enc_hidden_state.view(batch_size, -1, hidden_states.shape[-1])
                # 添加到新列表中
                new_encoder_hidden_states.append(enc_hidden_state)

            # 更新编码器隐藏状态列表
            encoder_hidden_states = new_encoder_hidden_states
            # 通过文本投影层进行投影
            T5_encoder_hidden_states = self.caption_projection[-1](T5_encoder_hidden_states)
            # 重塑张量
            T5_encoder_hidden_states = T5_encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])
            # 添加到编码器隐藏状态列表中
            encoder_hidden_states.append(T5_encoder_hidden_states)

        # 初始化文本 ID 张量
        txt_ids = torch.zeros(
            batch_size,
            encoder_hidden_states[-1].shape[1] + encoder_hidden_states[-2].shape[1] + encoder_hidden_states[0].shape[1],
            3,
            device=img_ids.device, dtype=img_ids.dtype
        )
        # 拼接图像和文本 ID
        ids = torch.cat((img_ids, txt_ids), dim=1) 
        # 通过 RoPE 嵌入器进行嵌入
        rope = self.pe_embedder(ids)

        # 2. Transformer 块处理
        # 块 ID 初始化
        block_id = 0
        # 拼接编码器隐藏状态
        initial_encoder_hidden_states = torch.cat([encoder_hidden_states[-1], encoder_hidden_states[-2]], dim=1)
        # 获取序列长度
        initial_encoder_hidden_states_seq_len = initial_encoder_hidden_states.shape[1]
        for bid, block in enumerate(self.double_stream_blocks):
            # 获取当前 LLaMA31 编码器隐藏状态
            cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id]
            # 拼接编码器隐藏状态
            cur_encoder_hidden_states = torch.cat([initial_encoder_hidden_states, cur_llama31_encoder_hidden_states], dim=1)
            # 应用 Transformer 块
            hidden_states, initial_encoder_hidden_states = block(
                image_tokens = hidden_states,
                image_tokens_masks = image_tokens_masks,
                text_tokens = cur_encoder_hidden_states,
                adaln_input = adaln_input,
                rope = rope,
            )
            # 截断初始编码器隐藏状态
            initial_encoder_hidden_states = initial_encoder_hidden_states[:, :initial_encoder_hidden_states_seq_len]
            # 增加块 ID
            block_id += 1

        # 获取图像标记的序列长度
        image_tokens_seq_len = hidden_states.shape[1]
        # 拼接隐藏状态
        hidden_states = torch.cat([hidden_states, initial_encoder_hidden_states], dim=1)
        # 获取隐藏状态的序列长度
        hidden_states_seq_len = hidden_states.shape[1]
        if image_tokens_masks is not None:
            # 创建编码器注意力掩码
            encoder_attention_mask_ones = torch.ones(
                (batch_size, initial_encoder_hidden_states.shape[1] + cur_llama31_encoder_hidden_states.shape[1]),
                device=image_tokens_masks.device, dtype=image_tokens_masks.dtype
            )
            # 拼接掩码
            image_tokens_masks = torch.cat([image_tokens_masks, encoder_attention_mask_ones], dim=1)

        for bid, block in enumerate(self.single_stream_blocks):
            # 获取当前 LLaMA31 编码器隐藏状态
            cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id]
            # 拼接隐藏状态
            hidden_states = torch.cat([hidden_states, cur_llama31_encoder_hidden_states], dim=1)
            # 应用 Transformer 块
            hidden_states = block(
                image_tokens=hidden_states,
                image_tokens_masks=image_tokens_masks,
                text_tokens=None,
                adaln_input=adaln_input,
                rope=rope,
            )
            # 截断隐藏状态
            hidden_states = hidden_states[:, :hidden_states_seq_len]
            # 增加块 ID
            block_id += 1

        # 截断隐藏状态
        hidden_states = hidden_states[:, :image_tokens_seq_len, ...]
        # 通过最后一层进行输出投影
        output = self.final_layer(hidden_states, adaln_input)
        # 重新排列小块图像为原始图像
        output = self.unpatchify(output, img_sizes)
        # 返回处理后的输出
        return -output[:, :, :h, :w]
