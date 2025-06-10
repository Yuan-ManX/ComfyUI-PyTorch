# original version: https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import torch
import torch.nn as nn
from einops import repeat

from modules.attention import optimized_attention
from flux.layers import EmbedND
from flux.math import apply_rope
import common_dit
import model_management


def sinusoidal_embedding_1d(dim, position):
    """
    生成1D正弦位置嵌入。
    
    Args:
        dim (int): 嵌入的维度，必须为偶数。
        position (Tensor): 位置张量，形状为 [..., L]。
    
    Returns:
        Tensor: 正弦位置嵌入，形状为 [..., L, dim]。
    """
    # 确保维度为偶数
    assert dim % 2 == 0
    # 计算一半的维度
    half = dim // 2

    # 将位置张量转换为浮点类型
    position = position.type(torch.float32)

    # 计算正弦和余弦项
    # torch.pow(10000, -torch.arange(half).to(position).div(half)) 生成频率项
    # torch.outer 计算外积，得到形状为 [L, half] 的张量
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    
    # 计算余弦和正弦，得到形状为 [L, dim] 的张量
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)

    # 返回位置嵌入，形状为 [..., L, dim]
    return x


class WanSelfAttention(nn.Module):
    """
    Wan 自注意力机制模块。
    
    Args:
        dim (int): 输入和输出的维度。
        num_heads (int): 多头注意力中的头数。
        window_size (tuple, optional): 窗口大小，默认为 (-1, -1)。
        qk_norm (bool, optional): 是否对查询和键进行归一化，默认为 True。
        eps (float, optional): RMSNorm 的小常数，默认为 1e-6。
        operation_settings (dict, optional): 操作设置，包含 'operations' 和其他配置，默认为 {}。
    """
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6, operation_settings={}):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # 定义查询、键、值和输出线性层
        self.q = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.k = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.v = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.o = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.norm_q = operation_settings.get("operations").RMSNorm(dim, eps=eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")) if qk_norm else nn.Identity()
        self.norm_k = operation_settings.get("operations").RMSNorm(dim, eps=eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")) if qk_norm else nn.Identity()

    def forward(self, x, freqs):
        """
        前向传播方法。
        
        Args:
            x (Tensor): 输入张量，形状为 [B, L, C]。
            freqs (Tensor): 旋转位置编码频率张量，形状为 [1024, C / num_heads / 2]。
        
        Returns:
            Tensor: 注意力输出，形状为 [B, L, C]。
        """
        # 获取张量的维度
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # 定义查询、键、值函数
        def qkv_fn(x):
            # 归一化并重塑查询张量
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            # 归一化并重塑键张量
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            # 重塑值张量
            v = self.v(x).view(b, s, n * d)
            return q, k, v

        # 获取查询、键、值张量
        q, k, v = qkv_fn(x)
        # 应用旋转位置编码
        q, k = apply_rope(q, k, freqs)

        # 执行多头自注意力机制
        x = optimized_attention(
            q.view(b, s, n * d),
            k.view(b, s, n * d),
            v,
            heads=self.num_heads,
        )

        # 应用输出线性层
        x = self.o(x)
        # 返回注意力输出
        return x


class WanT2VCrossAttention(WanSelfAttention):
    """
    Wan 文本到视频的跨注意力机制模块。
    
    Args:
        dim (int): 输入和输出的维度。
        num_heads (int): 多头注意力中的头数。
        window_size (tuple, optional): 窗口大小，默认为 (-1, -1)。
        qk_norm (bool, optional): 是否对查询和键进行归一化，默认为 True。
        eps (float, optional): RMSNorm 的小常数，默认为 1e-6。
        operation_settings (dict, optional): 操作设置，包含 'operations' 和其他配置，默认为 {}。
    """

    def forward(self, x, context, **kwargs):
        """
        前向传播方法，用于跨注意力机制。
        
        Args:
            x (Tensor): 输入张量，形状为 [B, L1, C]。
            context (Tensor): 上下文张量，形状为 [B, L2, C]。
            **kwargs: 其他可选参数。
        
        Returns:
            Tensor: 跨注意力输出，形状为 [B, L1, C]。
        """
        # 计算查询、键、值
        q = self.norm_q(self.q(x))  # 归一化并计算查询张量
        k = self.norm_k(self.k(context))  # 归一化并计算键张量
        v = self.v(context)  # 计算值张量

        # 执行多头跨注意力机制
        x = optimized_attention(q, k, v, heads=self.num_heads)

        # 应用输出线性层
        x = self.o(x)

        # 返回跨注意力输出
        return x


class WanI2VCrossAttention(WanSelfAttention):
    """
    Wan 图像到视频的跨注意力机制模块。
    
    Args:
        dim (int): 输入和输出的维度。
        num_heads (int): 多头注意力中的头数。
        window_size (tuple, optional): 窗口大小，默认为 (-1, -1)。
        qk_norm (bool, optional): 是否对查询和键进行归一化，默认为 True。
        eps (float, optional): RMSNorm 的小常数，默认为 1e-6。
        operation_settings (dict, optional): 操作设置，包含 'operations' 和其他配置，默认为 {}。
    """
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6, operation_settings={}):
        super().__init__(dim, num_heads, window_size, qk_norm, eps, operation_settings=operation_settings)

        # 定义用于图像的键和值线性层
        self.k_img = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.v_img = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = operation_settings.get("operations").RMSNorm(dim, eps=eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")) if qk_norm else nn.Identity()

    def forward(self, x, context, context_img_len):
        """
        前向传播方法，用于图像到视频的跨注意力机制。
        
        Args:
            x (Tensor): 输入张量，形状为 [B, L1, C]。
            context (Tensor): 上下文张量，形状为 [B, L2, C]。
            context_img_len (int): 图像上下文的长度。
        
        Returns:
            Tensor: 跨注意力输出，形状为 [B, L1, C]。
        """
        # 分离图像上下文和文本上下文
        context_img = context[:, :context_img_len]
        context = context[:, context_img_len:]

        # 计算查询、键、值
        q = self.norm_q(self.q(x))  # 归一化并计算查询张量
        k = self.norm_k(self.k(context))  # 归一化并计算键张量
        v = self.v(context)  # 计算值张量
        k_img = self.norm_k_img(self.k_img(context_img))  # 归一化并计算图像键张量
        v_img = self.v_img(context_img)  # 计算图像值张量

        # 执行图像到视频的跨注意力机制
        img_x = optimized_attention(q, k_img, v_img, heads=self.num_heads)
        
        # 执行文本到视频的跨注意力机制
        x = optimized_attention(q, k, v, heads=self.num_heads)

        # 结合图像和文本的注意力输出
        x = x + img_x  # 加权求和
        x = self.o(x)  # 应用输出线性层
        return x  # 返回跨注意力输出


# WAN_CROSSATTENTION_CLASSES 字典，映射跨注意力类型到相应的类
WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):
    """
    Wan 注意力块模块，结合自注意力和跨注意力机制。
    
    Args:
        cross_attn_type (str): 跨注意力类型，'t2v_cross_attn' 或 'i2v_cross_attn'。
        dim (int): 输入和输出的维度。
        ffn_dim (int): 前馈神经网络（FFN）的维度。
        num_heads (int): 多头注意力中的头数。
        window_size (tuple, optional): 窗口大小，默认为 (-1, -1)。
        qk_norm (bool, optional): 是否对查询和键进行归一化，默认为 True。
        cross_attn_norm (bool, optional): 是否对跨注意力进行归一化，默认为 False。
        eps (float, optional): RMSNorm 的小常数，默认为 1e-6。
        operation_settings (dict, optional): 操作设置，包含 'operations' 和其他配置，默认为 {}。
    """
    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6, operation_settings={}):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = operation_settings.get("operations").LayerNorm(dim, eps, elementwise_affine=False, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps, operation_settings=operation_settings)
        self.norm3 = operation_settings.get("operations").LayerNorm(
            dim, eps,
            elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps, operation_settings=operation_settings)
        self.norm2 = operation_settings.get("operations").LayerNorm(dim, eps, elementwise_affine=False, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.ffn = nn.Sequential(
            operation_settings.get("operations").Linear(dim, ffn_dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")), nn.GELU(approximate='tanh'),
            operation_settings.get("operations").Linear(ffn_dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

        # 定义调制参数
        self.modulation = nn.Parameter(torch.empty(1, 6, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

    def forward(
        self,
        x,
        e,
        freqs,
        context,
        context_img_len=257,
    ):
        """
        前向传播方法。
        
        Args:
            x (Tensor): 输入张量，形状为 [B, L, C]。
            e (Tensor): 调制张量，形状为 [B, 6, C]。
            freqs (Tensor): 旋转位置编码频率张量，形状为 [1024, C / num_heads / 2]。
            context (Tensor): 上下文张量，形状为 [B, L2, C]。
            context_img_len (int, optional): 图像上下文的长度，默认为 257。
        
        Returns:
            Tensor: 输出张量，形状为 [B, L, C]。
        """
        # assert e.dtype == torch.float32

        # 将调制张量 e 与 modulation 参数相加，并拆分为 6 个部分
        e = (model_management.cast_to(self.modulation, dtype=x.dtype, device=x.device) + e).chunk(6, dim=1)
        # assert e[0].dtype == torch.float32

        # 自注意力机制
        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0],
            freqs)

        x = x + y * e[2]

        # 跨注意力机制和前馈神经网络
        x = x + self.cross_attn(self.norm3(x), context, context_img_len=context_img_len)
        y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
        x = x + y * e[5]
        return x


class VaceWanAttentionBlock(WanAttentionBlock):
    """
    Vace Wan 注意力块模块，扩展自 WanAttentionBlock。
    
    Args:
        cross_attn_type (str): 跨注意力类型，'t2v_cross_attn' 或 'i2v_cross_attn'。
        dim (int): 输入和输出的维度。
        ffn_dim (int): 前馈神经网络（FFN）的维度。
        num_heads (int): 多头注意力中的头数。
        window_size (tuple, optional): 窗口大小，默认为 (-1, -1)。
        qk_norm (bool, optional): 是否对查询和键进行归一化，默认为 True。
        cross_attn_norm (bool, optional): 是否对跨注意力进行归一化，默认为 False。
        eps (float, optional): RMSNorm 的小常数，默认为 1e-6。
        block_id (int, optional): 块ID，默认为 0。
        operation_settings (dict, optional): 操作设置，包含 'operations' 和其他配置，默认为 {}。
    """
    def __init__(
            self,
            cross_attn_type,
            dim,
            ffn_dim,
            num_heads,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=False,
            eps=1e-6,
            block_id=0,
            operation_settings={}
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps, operation_settings=operation_settings)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = operation_settings.get("operations").Linear(self.dim, self.dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.after_proj = operation_settings.get("operations").Linear(self.dim, self.dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))

    def forward(self, c, x, **kwargs):
        if self.block_id == 0:
            c = self.before_proj(c) + x
        c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        return c_skip, c


class WanCamAdapter(nn.Module):
    """
    WanCam 适配器模块，用于处理视频数据。
    
    Args:
        in_dim (int): 输入维度。
        out_dim (int): 输出维度。
        kernel_size (int): 卷积核大小。
        stride (int): 卷积步幅。
        num_residual_blocks (int, optional): 残差块的数量，默认为 1。
        operation_settings (dict, optional): 操作设置，包含 'operations' 和其他配置，默认为 {}。
    """
    def __init__(self, in_dim, out_dim, kernel_size, stride, num_residual_blocks=1, operation_settings={}):
        super(WanCamAdapter, self).__init__()

        # Pixel Unshuffle: 将空间维度缩小8倍
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=8)

        # 卷积操作: 将空间维度缩小2倍（无重叠）
        self.conv = operation_settings.get("operations").Conv2d(in_dim * 64, out_dim, kernel_size=kernel_size, stride=stride, padding=0, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))

        # 残差块用于特征提取
        self.residual_blocks = nn.Sequential(
            *[WanCamResidualBlock(out_dim, operation_settings = operation_settings) for _ in range(num_residual_blocks)]
        )

    def forward(self, x):
        # 重塑张量，将帧维度合并到批次中
        bs, c, f, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(bs * f, c, h, w)

        # Pixel Unshuffle 操作
        x_unshuffled = self.pixel_unshuffle(x)

        # 卷积操作
        x_conv = self.conv(x_unshuffled)

        # 使用残差块进行特征提取
        out = self.residual_blocks(x_conv)

        # 重塑张量，恢复原始的 bf 维度
        out = out.view(bs, f, out.size(1), out.size(2), out.size(3))

        # 重新排列维度顺序（如果需要），例如交换通道和特征帧
        out = out.permute(0, 2, 1, 3, 4)

        return out


class WanCamResidualBlock(nn.Module):
    """
    WanCam 残差块模块。
    
    Args:
        dim (int): 输入和输出的维度。
        operation_settings (dict, optional): 操作设置，包含 'operations' 和其他配置，默认为 {}。
    """
    def __init__(self, dim, operation_settings={}):
        super(WanCamResidualBlock, self).__init__()
        self.conv1 = operation_settings.get("operations").Conv2d(dim, dim, kernel_size=3, padding=1, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = operation_settings.get("operations").Conv2d(dim, dim, kernel_size=3, padding=1, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out


class Head(nn.Module):
    """
    Head 模块，用于将输入特征映射到输出维度，并进行调制。
    
    Args:
        dim (int): 输入特征的维度。
        out_dim (int): 输出特征的维度。
        patch_size (tuple or int): 补丁大小，用于计算输出维度。
        eps (float, optional): LayerNorm 的小常数，默认为 1e-6。
        operation_settings (dict, optional): 操作设置，包含 'operations' 和其他配置，默认为 {}。
    """
    def __init__(self, dim, out_dim, patch_size, eps=1e-6, operation_settings={}):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # 计算输出维度
        out_dim = math.prod(patch_size) * out_dim
        self.norm = operation_settings.get("operations").LayerNorm(dim, eps, elementwise_affine=False, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.head = operation_settings.get("operations").Linear(dim, out_dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))

        # 定义调制参数
        self.modulation = nn.Parameter(torch.empty(1, 2, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

    def forward(self, x, e):
        """
        前向传播方法。
        
        Args:
            x (Tensor): 输入张量，形状为 [B, L1, C]。
            e (Tensor): 调制张量，形状为 [B, C]。
        
        Returns:
            Tensor: 输出张量，形状为 [B, L1, out_dim]。
        """
        # 确保 e 的数据类型为浮点类型
        # assert e.dtype == torch.float32

        # 将调制张量与 modulation 参数相加，并拆分为 2 个部分
        e = (model_management.cast_to(self.modulation, dtype=x.dtype, device=x.device) + e.unsqueeze(1)).chunk(2, dim=1)
        
        # 应用调制并进行前向传播
        x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):
    """
    MLPProj 模块，用于投影图像嵌入并进行位置嵌入。
    
    Args:
        in_dim (int): 输入维度。
        out_dim (int): 输出维度。
        flf_pos_embed_token_number (int, optional): 位置嵌入的 token 数量，默认为 None。
        operation_settings (dict, optional): 操作设置，包含 'operations' 和其他配置，默认为 {}。
    """
    def __init__(self, in_dim, out_dim, flf_pos_embed_token_number=None, operation_settings={}):
        super().__init__()

        # 定义投影网络
        self.proj = torch.nn.Sequential(
            operation_settings.get("operations").LayerNorm(in_dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")), operation_settings.get("operations").Linear(in_dim, in_dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")),
            torch.nn.GELU(), operation_settings.get("operations").Linear(in_dim, out_dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")),
            operation_settings.get("operations").LayerNorm(out_dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

        # 如果提供了位置嵌入的 token 数量，则定义位置嵌入参数
        if flf_pos_embed_token_number is not None:
            self.emb_pos = nn.Parameter(torch.empty((1, flf_pos_embed_token_number, in_dim), device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))
        else:
            self.emb_pos = None

    def forward(self, image_embeds):
        """
        前向传播方法。
        
        Args:
            image_embeds (Tensor): 输入图像嵌入，形状为 [B, L, in_dim]。
        
        Returns:
            Tensor: 投影后的图像嵌入，形状为 [B, L, out_dim]。
        """
        if self.emb_pos is not None:
            # 将位置嵌入添加到输入图像嵌入的前面
            image_embeds = image_embeds[:, :self.emb_pos.shape[1]] + model_management.cast_to(self.emb_pos[:, :image_embeds.shape[1]], dtype=image_embeds.dtype, device=image_embeds.device)

        # 应用投影网络
        clip_extra_context_tokens = self.proj(image_embeds)
        # 返回投影后的图像嵌入
        return clip_extra_context_tokens


class WanModel(torch.nn.Module):
    """
    Wan 扩散模型骨干，支持文本生成视频（text-to-video）和图像生成视频（image-to-video）。

    该模型是一个基于 Transformer 的扩散模型，能够处理视频数据的生成任务。
    """

    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 flf_pos_embed_token_number=None,
                 image_model=None,
                 device=None,
                 dtype=None,
                 operations=None,
                 ):
        """
        初始化扩散模型骨干。

        Args:
            model_type (str, 可选, 默认值为 't2v'):
                模型变体 - 't2v'（文本生成视频）或 'i2v'（图像生成视频）
            patch_size (tuple, 可选, 默认值为 (1, 2, 2)):
                视频嵌入的 3D 补丁尺寸（时间补丁尺寸，高度补丁尺寸，宽度补丁尺寸）
            text_len (int, 可选, 默认值为 512):
                文本嵌入的固定长度
            in_dim (int, 可选, 默认值为 16):
                输入视频的通道数（C_in）
            dim (int, 可选, 默认值为 2048):
                Transformer 的隐藏维度
            ffn_dim (int, 可选, 默认值为 8192):
                前馈网络中的中间维度
            freq_dim (int, 可选, 默认值为 256):
                正弦时间嵌入的维度
            text_dim (int, 可选, 默认值为 4096):
                文本嵌入的输入维度
            out_dim (int, 可选, 默认值为 16):
                输出视频的通道数（C_out）
            num_heads (int, 可选, 默认值为 16):
                注意力头的数量
            num_layers (int, 可选, 默认值为 32):
                Transformer 块的数量
            window_size (tuple, 可选, 默认值为 (-1, -1)):
                局部注意力的窗口尺寸（-1 表示全局注意力）
            qk_norm (bool, 可选, 默认值为 True):
                启用查询/键归一化
            cross_attn_norm (bool, 可选, 默认值为 False):
                启用跨注意力归一化
            eps (float, 可选, 默认值为 1e-6):
                归一化层的 epsilon 值
        """
        super().__init__()
        self.dtype = dtype
        operation_settings = {"operations": operations, "device": device, "dtype": dtype}

        # 检查 model_type 是否有效
        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        # 模型参数赋值
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # 嵌入层
        self.patch_embedding = operations.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size, device=operation_settings.get("device"), dtype=torch.float32)
        self.text_embedding = nn.Sequential(
            operations.Linear(text_dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")), nn.GELU(approximate='tanh'),
            operations.Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

        # 时间嵌入层
        self.time_embedding = nn.Sequential(
            operations.Linear(freq_dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")), nn.SiLU(), operations.Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))
        self.time_projection = nn.Sequential(nn.SiLU(), operations.Linear(dim, dim * 6, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

        # Transformer 块
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps, operation_settings=operation_settings)
            for _ in range(num_layers)
        ])

        # 输出头
        self.head = Head(dim, out_dim, patch_size, eps, operation_settings=operation_settings)

        # ROPE (旋转位置编码) 嵌入器
        d = dim // num_heads
        self.rope_embedder = EmbedND(dim=d, theta=10000.0, axes_dim=[d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)])
        
        # 如果是图像生成视频模式，初始化图像嵌入层
        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim, flf_pos_embed_token_number=flf_pos_embed_token_number, operation_settings=operation_settings)
        else:
            self.img_emb = None

    def forward_orig(
        self,
        x,
        t,
        context,
        clip_fea=None,
        freqs=None,
        transformer_options={},
        **kwargs,
    ):
        """
        扩散模型的正向传播。

        Args:
            x (Tensor):
                输入视频张量列表，形状为 [B, C_in, F, H, W]
            t (Tensor):
                扩散时间步长张量，形状为 [B]
            context (List[Tensor]):
                文本嵌入列表，每个形状为 [B, L, C]
            clip_fea (Tensor, 可选):
                CLIP 图像特征，用于图像生成视频模式
            y (List[Tensor], 可选):
                条件视频输入，用于图像生成视频模式，形状与 x 相同

        Returns:
            List[Tensor]:
                去噪后的视频张量列表，原始输入形状为 [C_out, F, H / 8, W / 8]
        """
        # 视频嵌入
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # 时间嵌入
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # 文本嵌入
        context = self.text_embedding(context)

        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                # CLIP 图像特征嵌入
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]

        # 处理 Transformer 块中的替换操作
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len)
                    return out
                out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs}, {"original_block": block_wrap})
                x = out["img"]
            else:
                x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)

        # 输出头处理
        x = self.head(x, e)

        # 反补丁化
        x = self.unpatchify(x, grid_sizes)
        return x

    def forward(self, x, timestep, context, clip_fea=None, time_dim_concat=None, transformer_options={}, **kwargs):
        bs, c, t, h, w = x.shape
        x = common_dit.pad_to_patch_size(x, self.patch_size)

        patch_size = self.patch_size
        t_len = ((t + (patch_size[0] // 2)) // patch_size[0])
        h_len = ((h + (patch_size[1] // 2)) // patch_size[1])
        w_len = ((w + (patch_size[2] // 2)) // patch_size[2])

        if time_dim_concat is not None:
            time_dim_concat = common_dit.pad_to_patch_size(time_dim_concat, self.patch_size)
            x = torch.cat([x, time_dim_concat], dim=2)
            t_len = ((x.shape[2] + (patch_size[0] // 2)) // patch_size[0])

        img_ids = torch.zeros((t_len, h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(0, t_len - 1, steps=t_len, device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
        img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).reshape(1, -1, 1)
        img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).reshape(1, 1, -1)
        img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=bs)

        freqs = self.rope_embedder(img_ids).movedim(1, 2)
        return self.forward_orig(x, timestep, context, clip_fea=clip_fea, freqs=freqs, transformer_options=transformer_options, **kwargs)[:, :, :t, :h, :w]

    def unpatchify(self, x, grid_sizes):
        """
        反补丁化操作，将扁平化的张量恢复为原始的 3D 视频形状。

        Args:
            x (Tensor):
                扁平化后的张量，形状为 [B, N, C_out * F * H * W]
            grid_sizes (tuple):
                网格尺寸，用于确定反补丁化的形状

        Returns:
            Tensor:
                反补丁化后的张量，形状为 [B, C_out, F, H, W]
        """
        # 反转转置和扁平化操作

        c = self.out_dim
        u = x
        b = u.shape[0]
        u = u[:, :math.prod(grid_sizes)].view(b, *grid_sizes, *self.patch_size, c)
        u = torch.einsum('bfhwpqrc->bcfphqwr', u)
        u = u.reshape(b, c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])
        return u


class VaceWanModel(WanModel):
    """
    Wan 扩散模型骨干的扩展版本，支持文本生成视频（text-to-video）和图像生成视频（image-to-video）。
    该扩展版本引入了 Vace 层，用于增强模型的生成能力和稳定性。
    """

    def __init__(self,
                 model_type='vace',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 flf_pos_embed_token_number=None,
                 image_model=None,
                 vace_layers=None,
                 vace_in_dim=None,
                 device=None,
                 dtype=None,
                 operations=None,
                 ):
        """
        初始化 VaceWanModel。

        Args:
            model_type (str, 可选, 默认值为 'vace'):
                模型变体 - 'vace'（Vace 扩展模型）
            patch_size (tuple, 可选, 默认值为 (1, 2, 2)):
                视频嵌入的 3D 补丁尺寸（时间补丁尺寸，高度补丁尺寸，宽度补丁尺寸）
            text_len (int, 可选, 默认值为 512):
                文本嵌入的固定长度
            in_dim (int, 可选, 默认值为 16):
                输入视频的通道数（C_in）
            dim (int, 可选, 默认值为 2048):
                Transformer 的隐藏维度
            ffn_dim (int, 可选, 默认值为 8192):
                前馈网络中的中间维度
            freq_dim (int, 可选, 默认值为 256):
                正弦时间嵌入的维度
            text_dim (int, 可选, 默认值为 4096):
                文本嵌入的输入维度
            out_dim (int, 可选, 默认值为 16):
                输出视频的通道数（C_out）
            num_heads (int, 可选, 默认值为 16):
                注意力头的数量
            num_layers (int, 可选, 默认值为 32):
                Transformer 块的数量
            window_size (tuple, 可选, 默认值为 (-1, -1)):
                局部注意力的窗口尺寸（-1 表示全局注意力）
            qk_norm (bool, 可选, 默认值为 True):
                启用查询/键归一化
            cross_attn_norm (bool, 可选, 默认值为 True):
                启用跨注意力归一化
            eps (float, 可选, 默认值为 1e-6):
                归一化层的 epsilon 值
            vace_layers (int, 可选):
                Vace 层的数量
            vace_in_dim (int, 可选):
                Vace 层的输入维度
            device (torch.device, 可选):
                计算设备
            dtype (torch.dtype, 可选):
                数据类型
            operations (module, 可选):
                自定义操作模块，包含 Conv3d 和 Linear 等操作
        """

        super().__init__(model_type='t2v', patch_size=patch_size, text_len=text_len, in_dim=in_dim, dim=dim, ffn_dim=ffn_dim, freq_dim=freq_dim, text_dim=text_dim, out_dim=out_dim, num_heads=num_heads, num_layers=num_layers, window_size=window_size, qk_norm=qk_norm, cross_attn_norm=cross_attn_norm, eps=eps, flf_pos_embed_token_number=flf_pos_embed_token_number, image_model=image_model, device=device, dtype=dtype, operations=operations)
        operation_settings = {"operations": operations, "device": device, "dtype": dtype}

        # Vace 扩展部分
        if vace_layers is not None:
            self.vace_layers = vace_layers
            self.vace_in_dim = vace_in_dim
            # Vace Transformer 块
            self.vace_blocks = nn.ModuleList([
                VaceWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim, self.num_heads, self.window_size, self.qk_norm, self.cross_attn_norm, self.eps, block_id=i, operation_settings=operation_settings)
                for i in range(self.vace_layers)
            ])

            # Vace 层映射，用于确定每个 Vace 层对应的主 Transformer 块
            self.vace_layers_mapping = {i: n for n, i in enumerate(range(0, self.num_layers, self.num_layers // self.vace_layers))}
            # Vace 补丁嵌入层
            self.vace_patch_embedding = operations.Conv3d(
                self.vace_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size, device=device, dtype=torch.float32
            )

    def forward_orig(
        self,
        x,
        t,
        context,
        vace_context,
        vace_strength,
        clip_fea=None,
        freqs=None,
        transformer_options={},
        **kwargs,
    ):
        # 视频嵌入
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # 时间嵌入
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # 文本嵌入
        context = self.text_embedding(context)

        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                # CLIP 图像特征嵌入
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]
        
        # 处理 Vace 上下文
        orig_shape = list(vace_context.shape)
        # 调整维度顺序并重塑形状
        vace_context = vace_context.movedim(0, 1).reshape([-1] + orig_shape[2:])
        # Vace 补丁嵌入
        c = self.vace_patch_embedding(vace_context.float()).to(vace_context.dtype)
        # 扁平化并转置
        c = c.flatten(2).transpose(1, 2)
        # 分割成列表，每个元素对应一个视频样本
        c = list(c.split(orig_shape[0], dim=0))

        # 主 Transformer 块处理
        x_orig = x

        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len)
                    return out
                out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs}, {"original_block": block_wrap})
                x = out["img"]
            else:
                x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)

            ii = self.vace_layers_mapping.get(i, None)
            if ii is not None:
                # Vace 层处理
                for iii in range(len(c)):
                    c_skip, c[iii] = self.vace_blocks[ii](c[iii], x=x_orig, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)
                    # 将 Vace 层的输出与主 Transformer 块的输出结合
                    x += c_skip * vace_strength[iii]
                # 删除临时变量以节省内存
                del c_skip
        # 输出头处理
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x


class CameraWanModel(WanModel):
    """
    Wan 扩散模型骨干的扩展版本，支持文本生成视频（text-to-video）和图像生成视频（image-to-video）。
    该扩展版本引入了相机条件适配器（Control Adapter），用于根据相机条件调整模型输出。
    """

    def __init__(self,
                 model_type='camera',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 flf_pos_embed_token_number=None,
                 image_model=None,
                 in_dim_control_adapter=24,
                 device=None,
                 dtype=None,
                 operations=None,
                 ):
        """
        初始化 CameraWanModel。

        Args:
            model_type (str, 可选, 默认值为 'camera'):
                模型变体 - 'camera'（相机条件模型）
            patch_size (tuple, 可选, 默认值为 (1, 2, 2)):
                视频嵌入的 3D 补丁尺寸（时间补丁尺寸，高度补丁尺寸，宽度补丁尺寸）
            text_len (int, 可选, 默认值为 512):
                文本嵌入的固定长度
            in_dim (int, 可选, 默认值为 16):
                输入视频的通道数（C_in）
            dim (int, 可选, 默认值为 2048):
                Transformer 的隐藏维度
            ffn_dim (int, 可选, 默认值为 8192):
                前馈网络中的中间维度
            freq_dim (int, 可选, 默认值为 256):
                正弦时间嵌入的维度
            text_dim (int, 可选, 默认值为 4096):
                文本嵌入的输入维度
            out_dim (int, 可选, 默认值为 16):
                输出视频的通道数（C_out）
            num_heads (int, 可选, 默认值为 16):
                注意力头的数量
            num_layers (int, 可选, 默认值为 32):
                Transformer 块的数量
            window_size (tuple, 可选, 默认值为 (-1, -1)):
                局部注意力的窗口尺寸（-1 表示全局注意力）
            qk_norm (bool, 可选, 默认值为 True):
                启用查询/键归一化
            cross_attn_norm (bool, 可选, 默认值为 True):
                启用跨注意力归一化
            eps (float, 可选, 默认值为 1e-6):
                归一化层的 epsilon 值
            in_dim_control_adapter (int, 可选, 默认值为 24):
                控制适配器的输入维度
            device (torch.device, 可选):
                计算设备
            dtype (torch.dtype, 可选):
                数据类型
            operations (module, 可选):
                自定义操作模块，包含 Conv3d 和 Linear 等操作
        """

        super().__init__(model_type='i2v', patch_size=patch_size, text_len=text_len, in_dim=in_dim, dim=dim, ffn_dim=ffn_dim, freq_dim=freq_dim, text_dim=text_dim, out_dim=out_dim, num_heads=num_heads, num_layers=num_layers, window_size=window_size, qk_norm=qk_norm, cross_attn_norm=cross_attn_norm, eps=eps, flf_pos_embed_token_number=flf_pos_embed_token_number, image_model=image_model, device=device, dtype=dtype, operations=operations)
        operation_settings = {"operations": operations, "device": device, "dtype": dtype}

        # 控制适配器，用于处理相机条件
        self.control_adapter = WanCamAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:], operation_settings=operation_settings)


    def forward_orig(
        self,
        x,
        t,
        context,
        clip_fea=None,
        freqs=None,
        camera_conditions = None,
        transformer_options={},
        **kwargs,
    ):
        # 视频嵌入
        x = self.patch_embedding(x.float()).to(x.dtype)
        if self.control_adapter is not None and camera_conditions is not None:
            x_camera = self.control_adapter(camera_conditions).to(x.dtype)
            x = x + x_camera
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # 时间嵌入
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # 文本嵌入
        context = self.text_embedding(context)

        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                # CLIP 图像特征嵌入
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]

        # Transformer 块处理
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len)
                    return out
                out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs}, {"original_block": block_wrap})
                x = out["img"]
            else:
                x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)

        # 输出头处理
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x
