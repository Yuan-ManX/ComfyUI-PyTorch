# code adapted from: https://github.com/Stability-AI/stable-audio-tools

from modules.attention import optimized_attention
import typing as tp
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
import math
import ops


class FourierFeatures(nn.Module):
    """
    FourierFeatures 类，用于生成傅里叶特征。
    该类通过线性变换将输入特征映射到傅里叶特征空间，适用于生成周期性特征。
    """
    def __init__(self, in_features, out_features, std=1., dtype=None, device=None):
        """
        初始化 FourierFeatures 模块。

        参数:
            in_features (int): 输入特征的维度。
            out_features (int): 输出特征的维度，必须为偶数。
            std (float, 可选): 初始化权重的标准差，默认为 1.0。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
        """
        super().__init__()
        # 确保输出特征维度为偶数
        assert out_features % 2 == 0
        # 初始化权重参数，形状为 (out_features // 2, in_features)
        self.weight = nn.Parameter(torch.empty(
            [out_features // 2, in_features], dtype=dtype, device=device))  # 使用正态分布初始化权重

    def forward(self, input):
        """
        前向传播函数，生成傅里叶特征。

        参数:
            input (torch.Tensor): 输入张量，形状为 (batch_size, ..., in_features)。

        返回:
            torch.Tensor: 傅里叶特征张量，形状为 (batch_size, ..., out_features)。
        """
        # 将权重转置并转换为与输入相同的设备和数据类型
        f = 2 * math.pi * input @ ops.cast_to_input(self.weight.T, input)
        # 计算正弦和余弦，并沿着最后一个维度拼接，生成傅里叶特征
        return torch.cat([f.cos(), f.sin()], dim=-1)


# norms
class LayerNorm(nn.Module):
    """
    LayerNorm 类，实现层归一化。
    该类实现了带有可学习参数和无偏置的层归一化。
    """
    def __init__(self, dim, bias=False, fix_scale=False, dtype=None, device=None):
        """
        初始化 LayerNorm 模块。

        参数:
            dim (int): 输入张量的维度。
            bias (bool, 可选): 是否使用偏置，默认为 False。
            fix_scale (bool, 可选): 是否固定缩放因子，默认为 False。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
        """
        super().__init__()

        # 初始化缩放因子参数
        self.gamma = nn.Parameter(torch.empty(dim, dtype=dtype, device=device))

        if bias:
            # 初始化偏置参数
            self.beta = nn.Parameter(torch.empty(dim, dtype=dtype, device=device))
        else:
            # 如果不使用偏置，则设置为 None
            self.beta = None

    def forward(self, x):
        """
        前向传播函数，应用层归一化。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 归一化后的张量。
        """
        beta = self.beta
        if beta is not None:
            # 将偏置参数转换为与输入相同的设备和数据类型
            beta = ops.cast_to_input(beta, x)
        # 应用层归一化，使用预训练好的 gamma 和 beta 参数
        return F.layer_norm(x, x.shape[-1:], weight=ops.cast_to_input(self.gamma, x), bias=beta)


class GLU(nn.Module):
    """
    GLU (Gated Linear Unit) 类，实现门控线性单元。
    GLU 是一种门控机制，通过将输入分为两部分，一部分作为门控信号，另一部分作为实际输入，进行逐元素相乘。
    """
    def __init__(
        self,
        dim_in,
        dim_out,
        activation,
        use_conv = False,
        conv_kernel_size = 3,
        dtype=None,
        device=None,
        operations=None,
    ):
        """
        初始化 GLU 模块。

        参数:
            dim_in (int): 输入特征的维度。
            dim_out (int): 输出特征的维度。
            activation (nn.Module): 激活函数。
            use_conv (bool, 可选): 是否使用卷积层，默认为 False。
            conv_kernel_size (int, 可选): 卷积核大小，默认为 3。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
            operations (module, 可选): 包含自定义操作的模块，默认为 None。
        """
        super().__init__()
        # 设置激活函数
        self.act = activation
        # 根据是否使用卷积层，选择不同的线性层或卷积层
        self.proj = operations.Linear(dim_in, dim_out * 2, dtype=dtype, device=device) if not use_conv else operations.Conv1d(dim_in, dim_out * 2, conv_kernel_size, padding = (conv_kernel_size // 2), dtype=dtype, device=device)
        # 标记是否使用卷积层
        self.use_conv = use_conv

    def forward(self, x):
        """
        前向传播函数，应用 GLU 激活函数。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 应用了 GLU 后的输出张量。
        """
        if self.use_conv:
            # 如果使用卷积层，则重塑张量以适应卷积操作
            x = rearrange(x, 'b n d -> b d n')
            x = self.proj(x)
            x = rearrange(x, 'b d n -> b n d')
        else:
            # 否则，使用线性层
            x = self.proj(x)

        # 将输出张量拆分为两个部分：激活部分和门控部分
        x, gate = x.chunk(2, dim = -1)
        # 应用激活函数到门控部分，并将其与激活部分相乘
        return x * self.act(gate)


class AbsolutePositionalEmbedding(nn.Module):
    """
    AbsolutePositionalEmbedding 类，实现绝对位置嵌入。
    该类通过嵌入层生成绝对位置嵌入，适用于序列数据。
    """
    def __init__(self, dim, max_seq_len):
        """
        初始化 AbsolutePositionalEmbedding 模块。

        参数:
            dim (int): 嵌入的维度。
            max_seq_len (int): 序列的最大长度。
        """
        super().__init__()
        # 计算缩放因子
        self.scale = dim ** -0.5
        # 设置最大序列长度
        self.max_seq_len = max_seq_len
        # 初始化嵌入层，嵌入维度为 dim，最大序列长度为 max_seq_len
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos = None, seq_start_pos = None):
        """
        前向传播函数，生成绝对位置嵌入。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, ...)。
            pos (torch.Tensor, 可选): 位置索引张量，默认为 None。
            seq_start_pos (torch.Tensor, 可选): 序列起始位置，默认为 None。

        返回:
            torch.Tensor: 绝对位置嵌入张量，形状为 (batch_size, seq_len, dim)。
        """
        # 获取序列长度和设备
        seq_len, device = x.shape[1], x.device
        # 检查序列长度是否超过最大长度
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if pos is None:
            # 如果未提供位置索引，则生成位置索引
            pos = torch.arange(seq_len, device = device)

        if seq_start_pos is not None:
            # 如果提供了序列起始位置，则调整位置索引
            pos = (pos - seq_start_pos[..., None]).clamp(min = 0)

        # 生成位置嵌入
        pos_emb = self.emb(pos)
        # 应用缩放因子
        pos_emb = pos_emb * self.scale
        return pos_emb


class ScaledSinusoidalEmbedding(nn.Module):
    """
    ScaledSinusoidalEmbedding 类，实现缩放的正弦位置嵌入。
    该类实现了缩放的正弦位置嵌入，适用于序列数据。
    """
    def __init__(self, dim, theta = 10000):
        """
        初始化 ScaledSinusoidalEmbedding 模块。

        参数:
            dim (int): 嵌入的维度，必须为偶数。
            theta (int, 可选): 控制频率的参数，默认为 10000。
        """
        super().__init__()
        # 确保维度为偶数
        assert (dim % 2) == 0, 'dimension must be divisible by 2'
        # 初始化缩放因子参数
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)

        # 计算维度的一半
        half_dim = dim // 2
        # 生成频率序列
        freq_seq = torch.arange(half_dim).float() / half_dim
        # 计算逆频率
        inv_freq = theta ** -freq_seq
        # 注册逆频率为缓冲区
        self.register_buffer('inv_freq', inv_freq, persistent = False)

    def forward(self, x, pos = None, seq_start_pos = None):
        """
        前向传播函数，生成缩放的正弦位置嵌入。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, ...)。
            pos (torch.Tensor, 可选): 位置索引张量，默认为 None。
            seq_start_pos (torch.Tensor, 可选): 序列起始位置，默认为 None。

        返回:
            torch.Tensor: 缩放的正弦位置嵌入张量，形状为 (batch_size, seq_len, dim)。
        """
        # 获取序列长度和设备
        seq_len, device = x.shape[1], x.device

        if pos is None:
            # 如果未提供位置索引，则生成位置索引
            pos = torch.arange(seq_len, device = device)

        if seq_start_pos is not None:
            # 如果提供了序列起始位置，则调整位置索引
            pos = pos - seq_start_pos[..., None]
        
        # 计算嵌入
        emb = torch.einsum('i, j -> i j', pos, self.inv_freq)
        # 计算正弦和余弦，并拼接
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        # 应用缩放因子
        return emb * self.scale


class RotaryEmbedding(nn.Module):
    """
    RotaryEmbedding 类，实现旋转位置编码（RoPE）。
    RoPE 是一种位置编码方法，通过旋转位置嵌入来引入位置信息，适用于 Transformer 模型。
    """
    def __init__(
        self,
        dim,
        use_xpos = False,
        scale_base = 512,
        interpolation_factor = 1.,
        base = 10000,
        base_rescale_factor = 1.,
        dtype=None,
        device=None,
    ):
        """
        初始化 RotaryEmbedding 模块。

        参数:
            dim (int): 嵌入的维度。
            use_xpos (bool, 可选): 是否使用 XPos 位置编码，默认为 False。
            scale_base (int, 可选): 缩放基，默认为 512。
            interpolation_factor (float, 可选): 插值因子，默认为 1.0。
            base (int, 可选): 基础参数，默认为 10000。
            base_rescale_factor (float, 可选): 基础重缩放因子，默认为 1.0。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
        """
        super().__init__()

        # 调整基础参数
        base *= base_rescale_factor ** (dim / (dim - 2))

        # 计算逆频率 inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', torch.empty((dim // 2,), device=device, dtype=dtype))

        # 确保插值因子大于或等于 1
        assert interpolation_factor >= 1.
        # 设置插值因子
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            # 如果不使用 XPos，则将 scale 设置为 None
            self.register_buffer('scale', None)
            return

        # 如果使用 XPos，则计算缩放因子 scale
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        # 设置缩放基
        self.scale_base = scale_base
        # 注册 scale 为缓冲区
        self.register_buffer('scale', scale)

    def forward_from_seq_len(self, seq_len, device, dtype):
        """
        从序列长度生成 RoPE。

        参数:
            seq_len (int): 序列长度。
            device (torch.device): 设备。
            dtype (torch.dtype): 数据类型。

        返回:
            tuple: 包含 RoPE 频率和缩放因子的元组。
        """
        # 生成位置索引张量
        t = torch.arange(seq_len, device=device, dtype=dtype)
        # 调用前向传播函数
        return self.forward(t)

    def forward(self, seq_len, t):
        """
        前向传播函数，生成 RoPE。

        参数:
            seq_len (int): 序列长度。
            t (torch.Tensor): 输入张量，形状为 (batch_size, seq_len)。

        返回:
            tuple: 包含 RoPE 频率和缩放因子的元组。
        """
        # device = self.inv_freq.device
        # 获取设备
        device = t.device

        # t = t.to(torch.float32)

        # 应用插值因子
        t = t / self.interpolation_factor

        # 计算频率
        freqs = torch.einsum('i , j -> i j', t, ops.cast_to_input(self.inv_freq, t))
        # 将频率重复一次并拼接
        freqs = torch.cat((freqs, freqs), dim = -1)

        if self.scale is None:
            # 如果不使用 XPos，则返回频率和 1 作为缩放因子
            return freqs, 1.

        # 计算缩放因子
        power = (torch.arange(seq_len, device = device) - (seq_len // 2)) / self.scale_base  # noqa: F821 seq_len is not defined
        scale = ops.cast_to_input(self.scale, t) ** rearrange(power, 'n -> n 1')
        # 将缩放因子重复一次并拼接
        scale = torch.cat((scale, scale), dim = -1)

        # 返回频率和缩放因子
        return freqs, scale


def rotate_half(x):
    """
    旋转张量的一半。

    参数:
        x (torch.Tensor): 输入张量，形状为 (..., 2, dim)。

    返回:
        torch.Tensor: 旋转后的张量，形状为 (..., 2, dim)。
    """
    # 重塑张量以分离两个部分
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    # 分离两个部分
    x1, x2 = x.unbind(dim = -2)
    # 旋转并拼接
    return torch.cat((-x2, x1), dim = -1)


def apply_rotary_pos_emb(t, freqs, scale = 1):
    """
    应用旋转位置嵌入（RoPE）到输入张量。

    参数:
        t (torch.Tensor): 输入张量，形状为 (..., seq_len, dim)。
        freqs (torch.Tensor): 频率张量，形状为 (seq_len, dim)。
        scale (float, 可选): 缩放因子，默认为 1。

    返回:
        torch.Tensor: 应用了 RoPE 后的张量，形状与输入相同。
    """
    # 保存输出数据类型
    out_dtype = t.dtype

    # 如果需要，为了数值稳定性，将张量转换为 float32
    # 计算输出数据类型
    dtype = t.dtype #reduce(torch.promote_types, (t.dtype, freqs.dtype, torch.float32))
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    freqs, t = freqs.to(dtype), t.to(dtype)
    # 截取与输入张量长度相同的频率
    freqs = freqs[-seq_len:, :]

    if t.ndim == 4 and freqs.ndim == 3:
        # 重塑频率张量以匹配输入张量
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    # 部分旋转位置嵌入，Wang 等人提出的 GPT-J 方法
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]

    # 应用 RoPE
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)

    # 转换回原始数据类型
    t, t_unrotated = t.to(out_dtype), t_unrotated.to(out_dtype)
    
    # 连接旋转后的张量和未旋转的张量
    return torch.cat((t, t_unrotated), dim = -1)


class FeedForward(nn.Module):
    """
    FeedForward 类，实现前馈神经网络（Feed-Forward Network）。
    该类实现了前馈神经网络，通常用于 Transformer 模型中的前馈层。
    """
    def __init__(
        self,
        dim,
        dim_out = None,
        mult = 4,
        no_bias = False,
        glu = True,
        use_conv = False,
        conv_kernel_size = 3,
        zero_init_output = True,
        dtype=None,
        device=None,
        operations=None,
    ):
        """
        初始化 FeedForward 模块。

        参数:
            dim (int): 输入特征的维度。
            dim_out (int, 可选): 输出特征的维度，默认为 None。
            mult (int, 可选): 隐藏层维度的倍数，默认为 4。
            no_bias (bool, 可选): 是否不使用偏置，默认为 False。
            glu (bool, 可选): 是否使用 GLU 激活函数，默认为 True。
            use_conv (bool, 可选): 是否使用卷积层，默认为 False。
            conv_kernel_size (int, 可选): 卷积核大小，默认为 3。
            zero_init_output (bool, 可选): 是否将输出层的权重初始化为零，默认为 True。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
            operations (module, 可选): 包含自定义操作的模块，默认为 None。
        """
        super().__init__()
        # 计算隐藏层维度
        inner_dim = int(dim * mult)

        # 默认使用 SwiGLU 激活函数
        activation = nn.SiLU()

        # 设置输出维度
        dim_out = dim if dim_out is None else dim_out

        if glu:
            # 如果使用 GLU 激活函数，则使用 GLU 模块
            linear_in = GLU(dim, inner_dim, activation, dtype=dtype, device=device, operations=operations)
        else:
            # 否则，使用线性层和激活函数的序列
            linear_in = nn.Sequential(
                rearrange('b n d -> b d n') if use_conv else nn.Identity(), # 如果使用卷积层，则重塑张量
                operations.Linear(dim, inner_dim, bias = not no_bias, dtype=dtype, device=device) if not use_conv else operations.Conv1d(dim, inner_dim, conv_kernel_size, padding = (conv_kernel_size // 2), bias = not no_bias, dtype=dtype, device=device),
                rearrange('b n d -> b d n') if use_conv else nn.Identity(), # 如果使用卷积层，则重塑张量
                activation  # 应用激活函数
            )

        # 定义输出线性层
        linear_out = operations.Linear(inner_dim, dim_out, bias = not no_bias, dtype=dtype, device=device) if not use_conv else operations.Conv1d(inner_dim, dim_out, conv_kernel_size, padding = (conv_kernel_size // 2), bias = not no_bias, dtype=dtype, device=device)

        # # init last linear layer to 0
        # if zero_init_output:
        #     nn.init.zeros_(linear_out.weight)
        #     if not no_bias:
        #         nn.init.zeros_(linear_out.bias)

        # 定义前馈神经网络的前馈部分
        self.ff = nn.Sequential(
            linear_in,  # 输入线性层或 GLU 模块
            rearrange('b d n -> b n d') if use_conv else nn.Identity(), # 如果使用卷积层，则重塑张量
            linear_out, # 输出线性层
            rearrange('b n d -> b d n') if use_conv else nn.Identity(), # 如果使用卷积层，则重塑张量
        )

    def forward(self, x):
        """
        前向传播函数，应用前馈神经网络。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 前馈神经网络的输出张量。
        """
        # 应用前馈部分并返回结果
        return self.ff(x)


class Attention(nn.Module):
    """
    Attention 类，实现多头自注意力机制。
    该类实现了多头自注意力机制，支持多种配置选项，如因果掩码、归一化等。
    """
    def __init__(
        self,
        dim,
        dim_heads = 64,
        dim_context = None,
        causal = False,
        zero_init_output=True,
        qk_norm = False,
        natten_kernel_size = None,
        dtype=None,
        device=None,
        operations=None,
    ):
        """
        初始化 Attention 模块。

        参数:
            dim (int): 注意力机制的维度。
            dim_heads (int, 可选): 每个头的维度，默认为 64。
            dim_context (int, 可选): 上下文特征的维度，默认为 None。
            causal (bool, 可选): 是否使用因果掩码，默认为 False。
            zero_init_output (bool, 可选): 是否将输出层的权重初始化为零，默认为 True。
            qk_norm (bool, 可选): 是否对查询和键进行归一化，默认为 False。
            natten_kernel_size (int, 可选): 自然注意力核大小，默认为 None。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
            operations (module, 可选): 包含自定义操作的模块，默认为 None。
        """
        super().__init__()
        # 设置注意力机制的维度
        self.dim = dim
        # 设置每个头的维度
        self.dim_heads = dim_heads
        # 设置是否使用因果掩码
        self.causal = causal
        
        # 设置键和值特征的维度
        dim_kv = dim_context if dim_context is not None else dim

        # 计算头的数量
        self.num_heads = dim // dim_heads
        # 计算键和值头的数量
        self.kv_heads = dim_kv // dim_heads

        if dim_context is not None:
            # 如果提供了上下文特征维度，则使用单独的线性层进行查询和键/值的投影
            self.to_q = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
            self.to_kv = operations.Linear(dim_kv, dim_kv * 2, bias=False, dtype=dtype, device=device)
        else:
            # 否则，使用融合的线性层进行查询、键和值的投影
            self.to_qkv = operations.Linear(dim, dim * 3, bias=False, dtype=dtype, device=device)

        # 定义输出线性层，将注意力机制的输出映射回原始维度
        self.to_out = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)

        # 如果需要将输出层的权重初始化为零，则执行初始化
        # if zero_init_output:
        #     nn.init.zeros_(self.to_out.weight)

        # 设置是否对查询和键进行归一化
        self.qk_norm = qk_norm

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        rotary_pos_emb = None,
        causal = None
    ):
        """
        前向传播函数，应用多头自注意力机制。

        参数:
            x (torch.Tensor): 输入张量。
            context (torch.Tensor, 可选): 上下文张量，默认为 None。
            mask (torch.Tensor, 可选): 注意力掩码，默认为 None。
            context_mask (torch.Tensor, 可选): 上下文掩码，默认为 None。
            rotary_pos_emb (torch.Tensor, 可选): 旋转位置嵌入，默认为 None。
            causal (bool, 可选): 因果掩码标志，默认为 None。

        返回:
            torch.Tensor: 注意力机制的输出张量。
        """
        # 获取头的数量和是否有上下文
        h, kv_h, has_context = self.num_heads, self.kv_heads, context is not None

        # 确定键和值输入
        kv_input = context if has_context else x

        if hasattr(self, 'to_q'):
            # Use separate linear projections for q and k/v
            # 如果存在单独的查询线性层，则使用它
            q = self.to_q(x)
            # 重塑查询张量以分离头的维度
            q = rearrange(q, 'b n (h d) -> b h n d', h = h)

            # 分割键和值，并重塑
            k, v = self.to_kv(kv_input).chunk(2, dim=-1)

            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = kv_h), (k, v))
        else:
            # Use fused linear projection
            # 否则，使用融合的线性层进行查询、键和值的投影，并重塑
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # 对查询和键进行归一化以进行余弦相似度注意力计算
        if self.qk_norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)

        if rotary_pos_emb is not None and not has_context:
            # 获取旋转位置嵌入
            freqs, _ = rotary_pos_emb

            # 保存查询和键的数据类型
            q_dtype = q.dtype
            k_dtype = k.dtype

            # 转换为 float32 以提高数值稳定性
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            freqs = freqs.to(torch.float32)

            # 应用 RoPE 到查询和键
            q = apply_rotary_pos_emb(q, freqs)
            k = apply_rotary_pos_emb(k, freqs)

            # 转换回原始数据类型
            q = q.to(q_dtype)
            k = k.to(k_dtype)

        # 设置输入掩码
        input_mask = context_mask

        if input_mask is None and not has_context:
            # 如果没有上下文掩码，则使用普通掩码
            input_mask = mask

        # 确定掩码
        masks = []

        if input_mask is not None:
            # 重塑掩码以适应注意力机制
            input_mask = rearrange(input_mask, 'b j -> b 1 1 j')
            # 添加掩码
            masks.append(~input_mask)

        # 获取序列长度
        n = q.shape[-2]

        # 确定因果掩码标志
        causal = self.causal if causal is None else causal

        if n == 1 and causal:
            # 如果序列长度为 1，则忽略因果掩码
            causal = False

        if h != kv_h:
            # 如果查询和键/值头的数量不匹配，则重复键/值头的数量以匹配查询头的数量
            heads_per_kv_head = h // kv_h
            k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim = 1), (k, v))

        # 调用优化的注意力函数
        out = optimized_attention(q, k, v, h, skip_reshape=True)
        # 应用输出线性层
        out = self.to_out(out)

        if mask is not None:
            # 重塑掩码以匹配输出张量
            mask = rearrange(mask, 'b n -> b n 1')
            # 应用掩码，填充为 0
            out = out.masked_fill(~mask, 0.)

        return out


class ConformerModule(nn.Module):
    """
    ConformerModule 类，实现了一个 Conformer 模块。
    Conformer 结合了卷积神经网络（CNN）和 Transformer 的优势，常用于语音识别和自然语言处理任务。
    该模块包括层归一化、逐点卷积、门控线性单元（GLU）、深度卷积、中间归一化和第二个逐点卷积。
    """
    def __init__(
        self,
        dim,
        norm_kwargs = {},
    ):
        """
        初始化 ConformerModule 模块。

        参数:
            dim (int): 输入和输出的维度。
            norm_kwargs (dict, 可选): LayerNorm 的关键字参数，默认为空字典。
        """
        super().__init__()

        # 设置维度
        self.dim = dim

        # 初始化层归一化层，使用传入的 norm_kwargs 参数
        self.in_norm = LayerNorm(dim, **norm_kwargs)
        # 初始化逐点卷积层，1x1 卷积，用于逐点特征变换
        self.pointwise_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        # 初始化 GLU 层，使用 SiLU 作为激活函数
        self.glu = GLU(dim, dim, nn.SiLU())
        # 初始化深度卷积层，核大小为 17，组数等于通道数，实现深度卷积
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=17, groups=dim, padding=8, bias=False)
        # 初始化中间层归一化层，使用传入的 norm_kwargs 参数
        self.mid_norm = LayerNorm(dim, **norm_kwargs) # This is a batch norm in the original but I don't like batch norm
        # 初始化 Swish 激活函数（也称为 SiLU）
        self.swish = nn.SiLU()
        # 初始化第二个逐点卷积层，1x1 卷积，用于逐点特征变换
        self.pointwise_conv_2 = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        """
        前向传播函数，应用 Conformer 模块的各个层。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, channels, seq_len)。

        返回:
            torch.Tensor: 输出张量，形状与输入相同。
        """
        # 应用层归一化
        x = self.in_norm(x)
        # 重塑张量以适应卷积层，从 (batch_size, channels, seq_len) 到 (batch_size, channels, seq_len)
        x = rearrange(x, 'b n d -> b d n')
        # 应用逐点卷积层
        x = self.pointwise_conv(x)
        # 重塑回原始形状
        x = rearrange(x, 'b d n -> b n d')
        # 应用 GLU 激活函数
        x = self.glu(x)
        # 重塑张量以适应深度卷积层
        x = rearrange(x, 'b n d -> b d n')
        # 应用深度卷积层
        x = self.depthwise_conv(x)
        # 重塑回原始形状
        x = rearrange(x, 'b d n -> b n d')
        # 应用中间层归一化
        x = self.mid_norm(x)
        # 应用 Swish 激活函数
        x = self.swish(x)
        # 重塑张量以适应第二个逐点卷积层
        x = rearrange(x, 'b n d -> b d n')
        # 应用第二个逐点卷积层
        x = self.pointwise_conv_2(x)
        # 重塑回原始形状
        x = rearrange(x, 'b d n -> b n d')

        # 返回输出张量
        return x


class TransformerBlock(nn.Module):
    """
    TransformerBlock 类，实现了一个 Transformer 模型中的基本模块。
    该模块包含自注意力机制、交叉注意力机制（可选）、前馈神经网络（Feed-Forward Network）以及可选的 Conformer 模块。
    """
    def __init__(
            self,
            dim,
            dim_heads = 64,
            cross_attend = False,
            dim_context = None,
            global_cond_dim = None,
            causal = False,
            zero_init_branch_outputs = True,
            conformer = False,
            layer_ix = -1,
            remove_norms = False,
            attn_kwargs = {},
            ff_kwargs = {},
            norm_kwargs = {},
            dtype=None,
            device=None,
            operations=None,
    ):
        """
        初始化 TransformerBlock 模块。

        参数:
            dim (int): 注意力机制的维度。
            dim_heads (int, 可选): 每个头的维度，默认为 64。
            cross_attend (bool, 可选): 是否使用交叉注意力，默认为 False。
            dim_context (int, 可选): 上下文特征的维度，默认为 None。
            global_cond_dim (int, 可选): 全局条件特征的维度，默认为 None。
            causal (bool, 可选): 是否使用因果掩码，默认为 False。
            zero_init_branch_outputs (bool, 可选): 是否将分支输出的权重初始化为零，默认为 True。
            conformer (bool, 可选): 是否使用 Conformer 模块，默认为 False。
            layer_ix (int, 可选): 当前层的索引，默认为 -1。
            remove_norms (bool, 可选): 是否移除归一化层，默认为 False。
            attn_kwargs (dict, 可选): 注意力机制的关键字参数，默认为空字典。
            ff_kwargs (dict, 可选): 前馈神经网络的关键字参数，默认为空字典。
            norm_kwargs (dict, 可选): 归一化的关键字参数，默认为空字典。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
            operations (module, 可选): 包含自定义操作的模块，默认为 None。
        """
        super().__init__()
        # 设置维度
        self.dim = dim
        # 设置每个头的维度
        self.dim_heads = dim_heads
        # 设置是否使用交叉注意力
        self.cross_attend = cross_attend
        # 设置上下文特征的维度
        self.dim_context = dim_context
        # 设置是否使用因果掩码
        self.causal = causal

        # 初始化预归一化层，如果 remove_norms 为 False，则使用 LayerNorm；否则，使用恒等函数
        self.pre_norm = LayerNorm(dim, dtype=dtype, device=device, **norm_kwargs) if not remove_norms else nn.Identity()

        # 初始化自注意力机制
        self.self_attn = Attention(
            dim,
            dim_heads = dim_heads,
            causal = causal,
            zero_init_output=zero_init_branch_outputs,
            dtype=dtype,
            device=device,
            operations=operations,
            **attn_kwargs
        )

        if cross_attend:
            # 如果使用交叉注意力，则初始化交叉注意力归一化层
            self.cross_attend_norm = LayerNorm(dim, dtype=dtype, device=device, **norm_kwargs) if not remove_norms else nn.Identity()
            # 初始化交叉注意力机制
            self.cross_attn = Attention(
                dim,
                dim_heads = dim_heads,
                dim_context=dim_context,
                causal = causal,
                zero_init_output=zero_init_branch_outputs,
                dtype=dtype,
                device=device,
                operations=operations,
                **attn_kwargs
            )

        # 初始化前馈神经网络的归一化层
        self.ff_norm = LayerNorm(dim, dtype=dtype, device=device, **norm_kwargs) if not remove_norms else nn.Identity()
        # 初始化前馈神经网络
        self.ff = FeedForward(dim, zero_init_output=zero_init_branch_outputs, dtype=dtype, device=device, operations=operations,**ff_kwargs)

        # 设置当前层的索引
        self.layer_ix = layer_ix

        # 如果使用 Conformer 模块，则初始化 Conformer 模块
        self.conformer = ConformerModule(dim, norm_kwargs=norm_kwargs) if conformer else None

        # 设置全局条件特征的维度
        self.global_cond_dim = global_cond_dim

        if global_cond_dim is not None:
            # 如果提供了全局条件特征维度，则初始化一个序列模块，用于生成缩放和平移因子
            self.to_scale_shift_gate = nn.Sequential(
                nn.SiLU(),
                nn.Linear(global_cond_dim, dim * 6, bias=False)
            )

            # 将线性层的权重初始化为零
            nn.init.zeros_(self.to_scale_shift_gate[1].weight)
            #nn.init.zeros_(self.to_scale_shift_gate_self[1].bias)

    def forward(
        self,
        x,
        context = None,
        global_cond=None,
        mask = None,
        context_mask = None,
        rotary_pos_emb = None
    ):
        """
        前向传播函数，应用 Transformer 模块的各个组件。

        参数:
            x (torch.Tensor): 输入张量。
            context (torch.Tensor, 可选): 上下文张量，默认为 None。
            global_cond (torch.Tensor, 可选): 全局条件张量，默认为 None。
            mask (torch.Tensor, 可选): 注意力掩码，默认为 None。
            context_mask (torch.Tensor, 可选): 上下文掩码，默认为 None。
            rotary_pos_emb (torch.Tensor, 可选): 旋转位置嵌入，默认为 None。

        返回:
            torch.Tensor: 输出张量。
        """
        if self.global_cond_dim is not None and self.global_cond_dim > 0 and global_cond is not None:

            # 如果提供了全局条件特征，则生成缩放和平移因子
            scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = self.to_scale_shift_gate(global_cond).unsqueeze(1).chunk(6, dim = -1)

            # self-attention with adaLN
            # 应用自注意力机制与自适应层归一化（adaLN）
            residual = x  # 保存残差
            x = self.pre_norm(x)  # 应用预归一化
            x = x * (1 + scale_self) + shift_self  # 应用缩放和平移
            x = self.self_attn(x, mask = mask, rotary_pos_emb = rotary_pos_emb)  # 应用自注意力机制
            x = x * torch.sigmoid(1 - gate_self)  # 应用门控机制
            x = x + residual  # 添加残差

            if context is not None:
                # 如果提供了上下文，则应用交叉注意力机制
                x = x + self.cross_attn(self.cross_attend_norm(x), context = context, context_mask = context_mask)

            if self.conformer is not None:
                # 如果使用了 Conformer 模块，则应用 Conformer 模块
                x = x + self.conformer(x)

            # 应用前馈神经网络与自适应层归一化（adaLN）
            residual = x
            x = self.ff_norm(x)
            # 应用缩放和平移
            x = x * (1 + scale_ff) + shift_ff
            # 应用前馈神经网络
            x = self.ff(x)
            # 应用门控机制
            x = x * torch.sigmoid(1 - gate_ff)
            # 添加残差
            x = x + residual

        else:
            # 如果没有提供全局条件特征，则应用自注意力机制
            x = x + self.self_attn(self.pre_norm(x), mask = mask, rotary_pos_emb = rotary_pos_emb)

            if context is not None:
                # 如果提供了上下文，则应用交叉注意力机制
                x = x + self.cross_attn(self.cross_attend_norm(x), context = context, context_mask = context_mask)

            if self.conformer is not None:
                # 如果使用了 Conformer 模块，则应用 Conformer 模块
                x = x + self.conformer(x)

            # 应用前馈神经网络
            x = x + self.ff(self.ff_norm(x))
        
        # 返回输出张量
        return x


class ContinuousTransformer(nn.Module):
    """
    ContinuousTransformer 类，实现了一个连续的 Transformer 模型。
    该模型支持多种嵌入方式（如旋转位置嵌入、正弦位置嵌入、绝对位置嵌入），
    以及可选的 Conformer 模块，适用于处理连续数据。
    """
    def __init__(
        self,
        dim,
        depth,
        *,
        dim_in = None,
        dim_out = None,
        dim_heads = 64,
        cross_attend=False,
        cond_token_dim=None,
        global_cond_dim=None,
        causal=False,
        rotary_pos_emb=True,
        zero_init_branch_outputs=True,
        conformer=False,
        use_sinusoidal_emb=False,
        use_abs_pos_emb=False,
        abs_pos_emb_max_length=10000,
        dtype=None,
        device=None,
        operations=None,
        **kwargs
        ):
        """
        初始化 ContinuousTransformer 模块。

        参数:
            dim (int): 模型的维度。
            depth (int): Transformer 层的数量。
            dim_in (int, 可选): 输入特征的维度，默认为 None。
            dim_out (int, 可选): 输出特征的维度，默认为 None。
            dim_heads (int, 可选): 每个头的维度，默认为 64。
            cross_attend (bool, 可选): 是否使用交叉注意力，默认为 False。
            cond_token_dim (int, 可选): 条件标记的维度，默认为 None。
            global_cond_dim (int, 可选): 全局条件特征的维度，默认为 None。
            causal (bool, 可选): 是否使用因果掩码，默认为 False。
            rotary_pos_emb (bool, 可选): 是否使用旋转位置嵌入，默认为 True。
            zero_init_branch_outputs (bool, 可选): 是否将分支输出的权重初始化为零，默认为 True。
            conformer (bool, 可选): 是否使用 Conformer 模块，默认为 False。
            use_sinusoidal_emb (bool, 可选): 是否使用正弦位置嵌入，默认为 False。
            use_abs_pos_emb (bool, 可选): 是否使用绝对位置嵌入，默认为 False。
            abs_pos_emb_max_length (int, 可选): 绝对位置嵌入的最大长度，默认为 10000。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
            operations (module, 可选): 包含自定义操作的模块，默认为 None。
            **kwargs: 其他关键字参数。
        """
        super().__init__()

        # 设置模型的维度
        self.dim = dim
        # 设置 Transformer 层的数量
        self.depth = depth
        # 设置是否使用因果掩码
        self.causal = causal
        # 初始化 Transformer 层列表
        self.layers = nn.ModuleList([])

        # 如果提供了输入特征的维度，则初始化输入投影层；否则，使用恒等函数
        self.project_in = operations.Linear(dim_in, dim, bias=False, dtype=dtype, device=device) if dim_in is not None else nn.Identity()
        # 如果提供了输出特征的维度，则初始化输出投影层；否则，使用恒等函数
        self.project_out = operations.Linear(dim, dim_out, bias=False, dtype=dtype, device=device) if dim_out is not None else nn.Identity()

        if rotary_pos_emb:
            # 如果使用旋转位置嵌入，则初始化旋转位置嵌入模块
            self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32), device=device, dtype=dtype)
        else:
            self.rotary_pos_emb = None

        # 设置是否使用正弦位置嵌入
        self.use_sinusoidal_emb = use_sinusoidal_emb
        if use_sinusoidal_emb:
            # 如果使用正弦位置嵌入，则初始化正弦位置嵌入模块
            self.pos_emb = ScaledSinusoidalEmbedding(dim)

        # 设置是否使用绝对位置嵌入
        self.use_abs_pos_emb = use_abs_pos_emb
        if use_abs_pos_emb:
            # 如果使用绝对位置嵌入，则初始化绝对位置嵌入模块
            self.pos_emb = AbsolutePositionalEmbedding(dim, abs_pos_emb_max_length)

        for i in range(depth):
            # 初始化 Transformer 层列表，每个层包含指定的参数
            self.layers.append(
                TransformerBlock(
                    dim,
                    dim_heads = dim_heads,
                    cross_attend = cross_attend,
                    dim_context = cond_token_dim,
                    global_cond_dim = global_cond_dim,
                    causal = causal,
                    zero_init_branch_outputs = zero_init_branch_outputs,
                    conformer=conformer,
                    layer_ix=i,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                    **kwargs
                )
            )

    def forward(
        self,
        x,
        mask = None,
        prepend_embeds = None,
        prepend_mask = None,
        global_cond = None,
        return_info = False,
        **kwargs
    ):
        """
        前向传播函数，处理输入数据并生成输出。

        参数:
            x (torch.Tensor): 输入张量。
            mask (torch.Tensor, 可选): 注意力掩码，默认为 None。
            prepend_embeds (torch.Tensor, 可选): 前置嵌入，默认为 None。
            prepend_mask (torch.Tensor, 可选): 前置掩码，默认为 None。
            global_cond (torch.Tensor, 可选): 全局条件张量，默认为 None。
            return_info (bool, 可选): 是否返回附加信息，默认为 False。
            **kwargs: 其他关键字参数。

        返回:
            torch.Tensor: 输出张量。
        """
        # 获取 Transformer 选项中的 "patches_replace" 部分
        patches_replace = kwargs.get("transformer_options", {}).get("patches_replace", {})
        # 获取批次大小、序列长度和设备信息
        batch, seq, device = *x.shape[:2], x.device
        # 获取上下文信息
        context = kwargs["context"]

        # 初始化隐藏状态列表
        info = {
            "hidden_states": [],
        }

        # 应用输入投影层
        x = self.project_in(x)

        if prepend_embeds is not None:
            # 获取前置嵌入的形状信息
            prepend_length, prepend_dim = prepend_embeds.shape[1:]

            # 确保前置嵌入的维度与序列维度匹配
            assert prepend_dim == x.shape[-1], 'prepend dimension must match sequence dimension'

            # 将前置嵌入与输入张量连接起来
            x = torch.cat((prepend_embeds, x), dim = -2)

            if prepend_mask is not None or mask is not None:
                # 如果未提供掩码，则初始化为全 1
                mask = mask if mask is not None else torch.ones((batch, seq), device = device, dtype = torch.bool)
                # 如果未提供前置掩码，则初始化为全 1
                prepend_mask = prepend_mask if prepend_mask is not None else torch.ones((batch, prepend_length), device = device, dtype = torch.bool)

                # 将前置掩码与掩码连接起来
                mask = torch.cat((prepend_mask, mask), dim = -1)

        # Attention layers

        if self.rotary_pos_emb is not None:
            # 如果使用了旋转位置嵌入，则生成旋转位置嵌入
            rotary_pos_emb = self.rotary_pos_emb.forward_from_seq_len(x.shape[1], dtype=x.dtype, device=x.device)
        else:
            rotary_pos_emb = None

        if self.use_sinusoidal_emb or self.use_abs_pos_emb:
            # 如果使用了正弦或绝对位置嵌入，则将位置嵌入添加到输入张量中
            x = x + self.pos_emb(x)

        # 获取 Transformer 选项中的 "dit" 部分
        blocks_replace = patches_replace.get("dit", {})
        # 遍历 Transformer 层列表
        for i, layer in enumerate(self.layers):
            # 如果当前双流块需要被替换，则执行替换操作
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = layer(args["img"], rotary_pos_emb=args["pe"], global_cond=args["vec"], context=args["txt"])
                    return out

                out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": global_cond, "pe": rotary_pos_emb}, {"original_block": block_wrap})
                x = out["img"]
            else:
                # 否则，正常执行 Transformer 层的前向传播
                x = layer(x, rotary_pos_emb = rotary_pos_emb, global_cond=global_cond, context=context)
            # x = checkpoint(layer, x, rotary_pos_emb = rotary_pos_emb, global_cond=global_cond, **kwargs)

            if return_info:
                # 如果需要返回附加信息，则将隐藏状态添加到列表中
                info["hidden_states"].append(x)
        
        # 应用输出投影层
        x = self.project_out(x)

        if return_info:
            # 如果需要返回附加信息，则返回输出张量和 info 字典
            return x, info

        # 否则，仅返回输出张量
        return x


class AudioDiffusionTransformer(nn.Module):
    """
    AudioDiffusionTransformer 类，实现了一个用于音频生成的扩散 Transformer 模型。
    该模型结合了变分自编码器（VAE）和 Transformer 的结构，支持多种条件嵌入和预处理步骤。
    """
    def __init__(self,
        io_channels=64,
        patch_size=1,
        embed_dim=1536,
        cond_token_dim=768,
        project_cond_tokens=False,
        global_cond_dim=1536,
        project_global_cond=True,
        input_concat_dim=0,
        prepend_cond_dim=0,
        depth=24,
        num_heads=24,
        transformer_type: tp.Literal["continuous_transformer"] = "continuous_transformer",
        global_cond_type: tp.Literal["prepend", "adaLN"] = "prepend",
        audio_model="",
        dtype=None,
        device=None,
        operations=None,
        **kwargs):
        """
        初始化 AudioDiffusionTransformer 模块。

        参数:
            io_channels (int, 可选): 输入和输出通道数，默认为 64。
            patch_size (int, 可选): 补丁大小，默认为 1。
            embed_dim (int, 可选): 嵌入维度，默认为 1536。
            cond_token_dim (int, 可选): 条件标记的维度，默认为 768。
            project_cond_tokens (bool, 可选): 是否对条件标记进行投影，默认为 False。
            global_cond_dim (int, 可选): 全局条件特征的维度，默认为 1536。
            project_global_cond (bool, 可选): 是否对全局条件特征进行投影，默认为 True。
            input_concat_dim (int, 可选): 输入连接特征的维度，默认为 0。
            prepend_cond_dim (int, 可选): 前置条件特征的维度，默认为 0。
            depth (int, 可选): Transformer 的深度，默认为 24。
            num_heads (int, 可选): 自注意力机制中头的数量，默认为 24。
            transformer_type (str, 可选): Transformer 的类型，默认为 "continuous_transformer"。
            global_cond_type (str, 可选): 全局条件类型，默认为 "prepend"。
            audio_model (str, 可选): 音频模型的名称，默认为空字符串。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
            operations (module, 可选): 包含自定义操作的模块，默认为 None。
            **kwargs: 其他关键字参数。
        """
        super().__init__()

        # 设置数据类型
        self.dtype = dtype
        # 设置条件标记的维度
        self.cond_token_dim = cond_token_dim

        # 时间步嵌入的维度，默认为 256
        timestep_features_dim = 256

        # 生成傅里叶特征
        self.timestep_features = FourierFeatures(1, timestep_features_dim, dtype=dtype, device=device)

        # 定义时间步嵌入的线性变换
        self.to_timestep_embed = nn.Sequential(
            operations.Linear(timestep_features_dim, embed_dim, bias=True, dtype=dtype, device=device), # 线性变换
            nn.SiLU(),
            operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device), # 线性变换
        )

        if cond_token_dim > 0:
            # 如果条件标记维度大于 0，则定义条件嵌入
            cond_embed_dim = cond_token_dim if not project_cond_tokens else embed_dim  # 计算条件嵌入的维度
            self.to_cond_embed = nn.Sequential(
                operations.Linear(cond_token_dim, cond_embed_dim, bias=False, dtype=dtype, device=device), # 线性变换
                nn.SiLU(),
                operations.Linear(cond_embed_dim, cond_embed_dim, bias=False, dtype=dtype, device=device) # 线性变换
            )
        else:
            cond_embed_dim = 0

        if global_cond_dim > 0:
            # 如果全局条件特征维度大于 0，则定义全局条件嵌入
            global_embed_dim = global_cond_dim if not project_global_cond else embed_dim  # 计算全局条件嵌入的维度
            self.to_global_embed = nn.Sequential(
                operations.Linear(global_cond_dim, global_embed_dim, bias=False, dtype=dtype, device=device), # 线性变换
                nn.SiLU(),
                operations.Linear(global_embed_dim, global_embed_dim, bias=False, dtype=dtype, device=device) # 线性变换
            )

        if prepend_cond_dim > 0:
            # 如果前置条件特征维度大于 0，则定义前置条件嵌入
            self.to_prepend_embed = nn.Sequential(
                operations.Linear(prepend_cond_dim, embed_dim, bias=False, dtype=dtype, device=device), # 线性变换
                nn.SiLU(),
                operations.Linear(embed_dim, embed_dim, bias=False, dtype=dtype, device=device) # 线性变换
            )

        # 设置输入连接特征的维度
        self.input_concat_dim = input_concat_dim

        # 计算输入维度
        dim_in = io_channels + self.input_concat_dim

        # 设置补丁大小
        self.patch_size = patch_size

        # Transformer
        # 设置 Transformer 类型
        self.transformer_type = transformer_type
        # 设置全局条件类型
        self.global_cond_type = global_cond_type

        if self.transformer_type == "continuous_transformer":
            # 初始化全局维度为 None
            global_dim = None

            if self.global_cond_type == "adaLN":
                # 如果全局条件类型为 adaLN，则将全局条件特征投影到嵌入维度
                global_dim = embed_dim

            # 初始化 ContinuousTransformer
            self.transformer = ContinuousTransformer(
                dim=embed_dim,
                depth=depth,
                dim_heads=embed_dim // num_heads,
                dim_in=dim_in * patch_size,
                dim_out=io_channels * patch_size,
                cross_attend = cond_token_dim > 0,
                cond_token_dim = cond_embed_dim,
                global_cond_dim=global_dim,
                dtype=dtype,
                device=device,
                operations=operations,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown transformer type: {self.transformer_type}")

        # 定义预处理卷积层
        self.preprocess_conv = operations.Conv1d(dim_in, dim_in, 1, bias=False, dtype=dtype, device=device)
        # 定义后处理卷积层
        self.postprocess_conv = operations.Conv1d(io_channels, io_channels, 1, bias=False, dtype=dtype, device=device)

    def _forward(
        self,
        x,
        t,
        mask=None,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        input_concat_cond=None,
        global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        return_info=False,
        **kwargs):
        """
        前向传播函数，处理输入数据并生成输出。

        参数:
            x (torch.Tensor): 输入张量。
            t (torch.Tensor): 时间步张量。
            mask (torch.Tensor, 可选): 注意力掩码，默认为 None。
            cross_attn_cond (torch.Tensor, 可选): 交叉注意力条件，默认为 None。
            cross_attn_cond_mask (torch.Tensor, 可选): 交叉注意力条件掩码，默认为 None。
            input_concat_cond (torch.Tensor, 可选): 输入连接条件，默认为 None。
            global_embed (torch.Tensor, 可选): 全局嵌入，默认为 None。
            prepend_cond (torch.Tensor, 可选): 前置条件，默认为 None。
            prepend_cond_mask (torch.Tensor, 可选): 前置条件掩码，默认为 None。
            return_info (bool, 可选): 是否返回附加信息，默认为 False。
            **kwargs: 其他关键字参数。

        返回:
            tuple: 包含输出张量和附加信息的元组。
        """
        if cross_attn_cond is not None:
            # 如果提供了交叉注意力条件，则对其进行嵌入
            cross_attn_cond = self.to_cond_embed(cross_attn_cond)

        if global_embed is not None:
            # 如果提供了全局嵌入，则对其进行嵌入
            global_embed = self.to_global_embed(global_embed)

        prepend_inputs = None
        prepend_mask = None
        prepend_length = 0
        if prepend_cond is not None:
            # 如果提供了前置条件，则对其进行嵌入
            prepend_cond = self.to_prepend_embed(prepend_cond)

            prepend_inputs = prepend_cond
            if prepend_cond_mask is not None:
                prepend_mask = prepend_cond_mask

        if input_concat_cond is not None:
            # 如果提供了输入连接条件，则对其进行插值以匹配输入张量的长度
            if input_concat_cond.shape[2] != x.shape[2]:
                input_concat_cond = F.interpolate(input_concat_cond, (x.shape[2], ), mode='nearest')
            # 将输入连接条件与输入张量连接起来
            x = torch.cat([x, input_concat_cond], dim=1)

        # 获取时间步嵌入
        timestep_embed = self.to_timestep_embed(self.timestep_features(t[:, None]).to(x.dtype)) # (b, embed_dim)

        # 将时间步嵌入视为全局嵌入，如果存在全局嵌入，则将其添加到全局嵌入中
        if global_embed is not None:
            global_embed = global_embed + timestep_embed
        else:
            global_embed = timestep_embed

        # 如果 Transformer 的全局条件类型为 "prepend"，则将全局嵌入作为前置输入
        if self.global_cond_type == "prepend":
            if prepend_inputs is None:
                # 如果没有前置输入，则将全局嵌入作为前置输入，前置掩码为全 1
                prepend_inputs = global_embed.unsqueeze(1)
                prepend_mask = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)
            else:
                # 如果有前置输入，则将全局嵌入与前置输入连接起来，前置掩码也进行连接
                prepend_inputs = torch.cat([prepend_inputs, global_embed.unsqueeze(1)], dim=1)
                prepend_mask = torch.cat([prepend_mask, torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)], dim=1)

            prepend_length = prepend_inputs.shape[1]

        # 应用预处理卷积层
        x = self.preprocess_conv(x) + x

        # 重塑张量以适应 Transformer 输入
        x = rearrange(x, "b c t -> b t c")

        extra_args = {}

        if self.global_cond_type == "adaLN":
            extra_args["global_cond"] = global_embed

        if self.patch_size > 1:
            # 如果补丁大小大于 1，则重塑张量以适应补丁大小
            x = rearrange(x, "b (t p) c -> b t (c p)", p=self.patch_size)

        # 根据 Transformer 类型调用不同的 Transformer 模块
        if self.transformer_type == "x-transformers":
            output = self.transformer(x, prepend_embeds=prepend_inputs, context=cross_attn_cond, context_mask=cross_attn_cond_mask, mask=mask, prepend_mask=prepend_mask, **extra_args, **kwargs)
        elif self.transformer_type == "continuous_transformer":
            output = self.transformer(x, prepend_embeds=prepend_inputs, context=cross_attn_cond, context_mask=cross_attn_cond_mask, mask=mask, prepend_mask=prepend_mask, return_info=return_info, **extra_args, **kwargs)

            if return_info:
                output, info = output
        elif self.transformer_type == "mm_transformer":
            output = self.transformer(x, context=cross_attn_cond, mask=mask, context_mask=cross_attn_cond_mask, **extra_args, **kwargs)

        # 重塑输出张量以恢复原始形状
        output = rearrange(output, "b t c -> b c t")[:,:,prepend_length:]

        if self.patch_size > 1:
            # 如果补丁大小大于 1，则重塑输出张量以恢复原始形状
            output = rearrange(output, "b (c p) t -> b c (t p)", p=self.patch_size)

        # 应用后处理卷积层
        output = self.postprocess_conv(output) + output

        if return_info:
            # 如果需要返回附加信息，则返回输出张量和 info
            return output, info

        return output

    def forward(
        self,
        x,
        timestep,
        context=None,
        context_mask=None,
        input_concat_cond=None,
        global_embed=None,
        negative_global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        mask=None,
        return_info=False,
        control=None,
        **kwargs):
            """
            前向传播函数，处理输入数据并生成输出。

            参数:
                x (torch.Tensor): 输入张量。
                timestep (torch.Tensor): 时间步张量。
                context (torch.Tensor, 可选): 交叉注意力条件，默认为 None。
                context_mask (torch.Tensor, 可选): 交叉注意力条件掩码，默认为 None。
                input_concat_cond (torch.Tensor, 可选): 输入连接条件，默认为 None。
                global_embed (torch.Tensor, 可选): 全局嵌入，默认为 None。
                negative_global_embed (torch.Tensor, 可选): 负全局嵌入，默认为 None。
                prepend_cond (torch.Tensor, 可选): 前置条件，默认为 None。
                prepend_cond_mask (torch.Tensor, 可选): 前置条件掩码，默认为 None。
                mask (torch.Tensor, 可选): 注意力掩码，默认为 None。
                return_info (bool, 可选): 是否返回附加信息，默认为 False。
                control (torch.Tensor, 可选): 控制信号，默认为 None。
                **kwargs: 其他关键字参数。

            返回:
                torch.Tensor: 输出张量。
            """
            return self._forward(
                x,
                timestep,
                cross_attn_cond=context,
                cross_attn_cond_mask=context_mask,
                input_concat_cond=input_concat_cond,
                global_embed=global_embed,
                prepend_cond=prepend_cond,
                prepend_cond_mask=prepend_cond_mask,
                mask=mask,
                return_info=return_info,
                **kwargs
            )
