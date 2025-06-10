# original version: https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/vae.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from modules.diffusionmodules.model import vae_attention

import ops
ops = ops.disable_weight_init

CACHE_T = 2


class CausalConv3d(ops.Conv3d):
    """
    CausalConv3d 类，实现了因果卷积（causal convolution）的 3D 卷积层。
    因果卷积确保输出在时间步 t 时的计算仅依赖于时间步 t 及之前的数据，
    这对于生成模型（如自回归模型）非常重要。
    """

    def __init__(self, *args, **kwargs):
        """
        初始化 CausalConv3d 模块。

        参数:
            *args: 传递给父类 Conv3d 的位置参数。
            **kwargs: 传递给父类 Conv3d 的关键字参数。
        """
        super().__init__(*args, **kwargs)
        # 设置填充参数，确保因果性
        # 填充顺序为 (前后时间步填充, 前后高度填充, 前后宽度填充)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        # 将填充参数重置为 (0, 0, 0)，因为填充将在前向传播中手动应用
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        """
        前向传播函数，应用因果卷积。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, channels, time_steps, height, width)。
            cache_x (torch.Tensor, 可选): 缓存的张量，用于保持时间步的因果性，默认为 None。

        返回:
            torch.Tensor: 输出张量，经过因果卷积处理后的结果。
        """
        # 将填充参数转换为列表，以便修改
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            # 如果提供了缓存张量并且时间步填充大于 0，则将缓存张量与输入张量连接起来
            cache_x = cache_x.to(x.device) # 将缓存张量移动到与输入张量相同的设备
            # 在时间步维度上连接缓存张量和输入张量
            x = torch.cat([cache_x, x], dim=2)
            # 调整时间步填充，以考虑缓存张量的长度
            padding[4] -= cache_x.shape[2]
        # 应用填充，确保因果性
        x = F.pad(x, padding)

        # 调用父类的前向传播方法，应用 3D 卷积
        return super().forward(x)


class RMS_norm(nn.Module):
    """
    RMS_norm 类，实现均方根归一化（RMS Normalization）。
    RMS 归一化是一种归一化技术，类似于层归一化（Layer Normalization），
    但使用均方根作为缩放因子。
    """
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        """
        初始化 RMS_norm 模块。

        参数:
            dim (int): 归一化的维度。
            channel_first (bool, 可选): 是否为通道优先，默认为 True。
            images (bool, 可选): 是否为图像数据，默认为 True。
            bias (bool, 可选): 是否使用偏置，默认为 False。
        """
        super().__init__()
        # 如果不是图像数据，则广播维度为 (1,)
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        # 如果是通道优先，则形状为 (dim, 1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        # 设置通道优先标志
        self.channel_first = channel_first
        # 计算缩放因子
        self.scale = dim**0.5
        # 初始化缩放参数 gamma
        self.gamma = nn.Parameter(torch.ones(shape))
        # 如果使用偏置，则初始化偏置参数
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else None

    def forward(self, x):
        """
        前向传播函数，应用 RMS 归一化。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 归一化后的张量。
        """
        # 应用 RMS 归一化
        return F.normalize(
            x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma.to(x) + (self.bias.to(x) if self.bias is not None else 0)


class Upsample(nn.Upsample):
    """
    Upsample 类，重载了 nn.Upsample 的前向传播方法，以修复 bfloat16 支持问题。
    """
    def forward(self, x):
        """
        前向传播函数，应用最近邻插值上采样。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 上采样后的张量。
        """
        # 将输入张量转换为 float 类型，应用上采样，再转换回原始数据类型
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
    """
    Resample 类，实现重采样操作，包括上采样和下采样。
    """
    def __init__(self, dim, mode):
        """
        初始化 Resample 模块。

        参数:
            dim (int): 输入和输出的维度。
            mode (str): 重采样的模式，可以是 'none', 'upsample2d', 'upsample3d', 'downsample2d', 'downsample3d'。
        """
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        # 设置维度
        self.dim = dim
        # 设置模式
        self.mode = mode

        # 根据模式定义重采样层
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),  # 最近邻上采样
                ops.Conv2d(dim, dim // 2, 3, padding=1))  # 卷积层
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),   # 最近邻上采样
                ops.Conv2d(dim, dim // 2, 3, padding=1))  # 卷积层
            self.time_conv = CausalConv3d(  # 因果 3D 卷积层
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),  # 零填充
                ops.Conv2d(dim, dim, 3, stride=(2, 2)))  # 卷积层，下采样
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),  # 零填充
                ops.Conv2d(dim, dim, 3, stride=(2, 2)))  # 卷积层，下采样
            self.time_conv = CausalConv3d(   # 因果 3D 卷积层
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()  # 如果模式为 'none'，则使用恒等函数

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        """
        前向传播函数，应用重采样操作。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, channels, time_steps, height, width)。
            feat_cache (torch.Tensor, 可选): 特征缓存，默认为 None。
            feat_idx (list of int, 可选): 特征索引列表，默认为 [0]。

        返回:
            torch.Tensor: 重采样后的张量。
        """
        # 获取输入张量的形状
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:
                    # 缓存最后 CACHE_T 个时间步的数据
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] != 'Rep':
                        # 如果缓存长度小于 2 且缓存不为 'Rep'，则将缓存与最后一张帧连接起来
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device), cache_x
                        ],
                                            dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] == 'Rep':
                        # 如果缓存长度小于 2 且缓存为 'Rep'，则用零张量填充
                        cache_x = torch.cat([
                            torch.zeros_like(cache_x).to(cache_x.device),
                            cache_x
                        ],
                                            dim=2)
                    if feat_cache[idx] == 'Rep':
                        # 应用因果 3D 卷积
                        x = self.time_conv(x)
                    else:
                        # 应用因果 3D 卷积，传入缓存
                        x = self.time_conv(x, feat_cache[idx])
                    # 更新缓存
                    feat_cache[idx] = cache_x
                    # 更新索引
                    feat_idx[0] += 1

                    # 重塑张量以分离两个时间步
                    x = x.reshape(b, 2, c, t, h, w)
                    # 堆叠时间步
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                    3)
                    # 重塑回原始形状
                    x = x.reshape(b, c, t * 2, h, w)
        # 获取时间步长度
        t = x.shape[2]
        # 重塑张量以适应重采样层
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        # 应用重采样层
        x = self.resample(x)
        # 重塑回原始形状
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    # 缓存输出张量
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    # 缓存最后一张帧
                    cache_x = x[:, :, -1:, :, :].clone()
                    
                    # if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx]!='Rep':
                    #     # cache last frame of last two chunk
                    #     cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

                    # 应用因果 3D 卷积，传入缓存
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    # 更新缓存
                    feat_cache[idx] = cache_x
                    # 更新索引
                    feat_idx[0] += 1
        # 返回重采样后的张量
        return x

    def init_weight(self, conv):
        """
        初始化卷积层的权重。

        参数:
            conv (nn.Conv3d): 要初始化的卷积层。
        """
        conv_weight = conv.weight
        # 将权重初始化为零
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        # 生成单位矩阵
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        #conv_weight.data[:,:,-1,1,1] = init_matrix * 0.5
        # 初始化特定位置的权重
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  #* 0.5
        # 将初始化后的权重复制回卷积层
        conv.weight.data.copy_(conv_weight)
        # 将偏置初始化为零
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        """
        初始化卷积层的权重。

        参数:
            conv (nn.Conv3d): 要初始化的卷积层。
        """
        conv_weight = conv.weight.data
        # 将权重初始化为零
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        # 生成单位矩阵
        init_matrix = torch.eye(c1 // 2, c2)
        #init_matrix = repeat(init_matrix, 'o ... -> (o 2) ...').permute(1,0,2).contiguous().reshape(c1,c2)
        # 初始化特定位置的权重
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        # 初始化特定位置的权重
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        # 将初始化后的权重复制回卷积层
        conv.weight.data.copy_(conv_weight)
        # 将偏置初始化为零
        nn.init.zeros_(conv.bias.data)


class ResidualBlock(nn.Module):
    """
    ResidualBlock 类，实现了一个残差块（Residual Block）。
    该残差块包含残差连接、归一化、激活函数、卷积层和 Dropout 层。
    """
    def __init__(self, in_dim, out_dim, dropout=0.0):
        """
        初始化 ResidualBlock 模块。

        参数:
            in_dim (int): 输入特征的维度。
            out_dim (int): 输出特征的维度。
            dropout (float, 可选): Dropout 概率，默认为 0.0。
        """
        super().__init__()
        # 设置输入维度
        self.in_dim = in_dim
        # 设置输出维度
        self.out_dim = out_dim

        # 定义残差路径的层序列
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),  # 均方根归一化
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),  # 因果 3D 卷积层
            CausalConv3d(out_dim, out_dim, 3, padding=1))  # 因果 3D 卷积层
        # 定义捷径路径的卷积层，如果输入和输出维度不同，则使用 1x1 卷积；否则，使用恒等函数
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        """
        前向传播函数，应用残差块。

        参数:
            x (torch.Tensor): 输入张量。
            feat_cache (torch.Tensor, 可选): 特征缓存，默认为 None。
            feat_idx (list of int, 可选): 特征索引列表，默认为 [0]。

        返回:
            torch.Tensor: 输出张量，经过残差块处理后的结果。
        """
        # 应用捷径路径
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                # 如果当前层是因果卷积层且提供了特征缓存，则处理缓存
                idx = feat_idx[0]
                # 缓存最后 CACHE_T 个时间步的数据
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                # 如果缓存长度小于 2 且缓存不为 None，则将缓存与最后一张帧连接起来
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                # 应用因果卷积层，传入缓存
                x = layer(x, feat_cache[idx])
                # 更新缓存
                feat_cache[idx] = cache_x
                # 更新索引
                feat_idx[0] += 1
            else:
                # 否则，正常应用层
                x = layer(x)
        # 将残差路径与捷径路径相加
        return x + h


class AttentionBlock(nn.Module):
    """
    AttentionBlock 类，实现了一个因果自注意力块（causal self-attention block）。
    该块包含归一化、查询-键-值（QKV）投影、优化的注意力机制和投影层。
    """

    def __init__(self, dim):
        """
        初始化 AttentionBlock 模块。

        参数:
            dim (int): 特征的维度。
        """
        super().__init__()
        # 设置维度
        self.dim = dim

        # 定义层序列
        self.norm = RMS_norm(dim)  # 均方根归一化
        self.to_qkv = ops.Conv2d(dim, dim * 3, 1)  # QKV 投影卷积层
        self.proj = ops.Conv2d(dim, dim, 1)  # 输出投影卷积层
        self.optimized_attention = vae_attention()  # 优化的注意力机制

    def forward(self, x):
        """
        前向传播函数，应用自注意力块。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 输出张量，经过自注意力块处理后的结果。
        """
        # 保存输入张量作为恒等映射
        identity = x
        # 获取输入张量的形状
        b, c, t, h, w = x.size()
        # 重塑张量以适应注意力机制
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        # 应用归一化
        x = self.norm(x)

        # 计算查询、键和值
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        # 应用优化的注意力机制
        x = self.optimized_attention(q, k, v)

        # 输出投影
        x = self.proj(x)
        # 重塑回原始形状
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        # 添加恒等映射
        return x + identity


class Encoder3d(nn.Module):
    """
    Encoder3d 类，实现了一个 3D 编码器。
    该编码器包含因果卷积、残差块、自注意力块、下采样层和输出头。
    """
    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        """
        初始化 Encoder3d 模块。

        参数:
            dim (int, 可选): 基础维度，默认为 128。
            z_dim (int, 可选): 潜在空间的维度，默认为 4。
            dim_mult (list of int, 可选): 每个下采样块的维度倍数列表，默认为 [1, 2, 4, 4]。
            num_res_blocks (int, 可选): 每个下采样块中残差块的数量，默认为 2。
            attn_scales (list of float, 可选): 自注意力机制的应用尺度列表，默认为空列表。
            temperal_downsample (list of bool, 可选): 是否进行时间步下采样，默认为 [True, True, False]。
            dropout (float, 可选): Dropout 概率，默认为 0.0。
        """
        super().__init__()
        self.dim = dim  # 设置基础维度
        self.z_dim = z_dim  # 设置潜在空间维度
        self.dim_mult = dim_mult  # 设置维度倍数列表
        self.num_res_blocks = num_res_blocks  # 设置残差块数量
        self.attn_scales = attn_scales  # 设置自注意力尺度列表
        self.temperal_downsample = temperal_downsample  # 设置时间步下采样标志列表

        # 计算每个块的维度
        dims = [dim * u for u in [1] + dim_mult]  # 计算每个块的维度
        scale = 1.0   # 初始化尺度因子

        # 初始化第一个卷积层，将输入图像从 3 通道映射到 dims[0] 通道
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # 初始化下采样块列表
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # 添加残差块和自注意力块
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # 添加下采样层
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[
                    i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        # 将下采样块组合成序列
        self.downsamples = nn.Sequential(*downsamples)

        # 初始化中间块，包括残差块和自注意力块
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout))

        # 初始化输出头，包括归一化、激活函数和因果卷积层
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        """
        前向传播函数，应用编码器。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, 3, time_steps, height, width)。
            feat_cache (torch.Tensor, 可选): 特征缓存，默认为 None。
            feat_idx (list of int, 可选): 特征索引列表，默认为 [0]。

        返回:
            torch.Tensor: 输出张量，经过编码器处理后的潜在空间。
        """
        if feat_cache is not None:
            idx = feat_idx[0]
            # 缓存最后 CACHE_T 个时间步的数据
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # 如果缓存长度小于 2 且缓存不为 None，则将缓存与最后一张帧连接起来
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            # 应用第一个因果卷积层，传入缓存
            x = self.conv1(x, feat_cache[idx])
            # 更新缓存
            feat_cache[idx] = cache_x
            # 更新索引
            feat_idx[0] += 1
        else:
            # 否则，正常应用第一个因果卷积层
            x = self.conv1(x)

        ## 下采样过程
        for layer in self.downsamples:
            if feat_cache is not None:
                # 应用下采样层，传入缓存
                x = layer(x, feat_cache, feat_idx)
            else:
                # 否则，正常应用下采样层
                x = layer(x)

        ## 中间过程
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                # 应用残差块，传入缓存
                x = layer(x, feat_cache, feat_idx)
            else:
                # 否则，正常应用层
                x = layer(x)

        ## 输出头
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                # 缓存最后 CACHE_T 个时间步的数据
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # 如果缓存长度小于 2 且缓存不为 None，则将缓存与最后一张帧连接起来
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                # 应用因果卷积层，传入缓存
                x = layer(x, feat_cache[idx])
                # 更新缓存
                feat_cache[idx] = cache_x
                # 更新索引
                feat_idx[0] += 1
            else:
                # 否则，正常应用层
                x = layer(x)
        return x


class Decoder3d(nn.Module):
    """
    Decoder3d 类，实现了一个 3D 解码器。
    该解码器包含因果卷积、残差块、自注意力块、上采样层和输出头。
    """
    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0):
        """
        初始化 Decoder3d 模块。

        参数:
            dim (int, 可选): 基础维度，默认为 128。
            z_dim (int, 可选): 潜在空间的维度，默认为 4。
            dim_mult (list of int, 可选): 每个上采样块的维度倍数列表，默认为 [1, 2, 4, 4]。
            num_res_blocks (int, 可选): 每个上采样块中残差块的数量，默认为 2。
            attn_scales (list of float, 可选): 自注意力机制的应用尺度列表，默认为空列表。
            temperal_upsample (list of bool, 可选): 是否进行时间步上采样，默认为 [False, True, True]。
            dropout (float, 可选): Dropout 概率，默认为 0.0。
        """
        super().__init__()
        self.dim = dim  # 设置基础维度
        self.z_dim = z_dim  # 设置潜在空间维度
        self.dim_mult = dim_mult  # 设置维度倍数列表
        self.num_res_blocks = num_res_blocks  # 设置残差块数量
        self.attn_scales = attn_scales  # 设置自注意力尺度列表
        self.temperal_upsample = temperal_upsample  # 设置时间步上采样标志列表

        # 计算每个块的维度
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        # 初始化尺度因子
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # 初始化第一个卷积层，将潜在空间从 z_dim 通道映射到 dims[0] 通道
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # 初始化中间块，包括残差块和自注意力块
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

        # 初始化上采样块列表
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # 如果是第二个、第三个或第四个块，则调整输入维度
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            # 添加残差块和自注意力块
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # 添加上采样层
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0

        # 将上采样块组合成序列
        self.upsamples = nn.Sequential(*upsamples)

        # 初始化输出头，包括均方根归一化、激活函数和因果卷积层
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        """
        前向传播函数，应用解码器。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, z_dim, time_steps, height, width)。
            feat_cache (torch.Tensor, 可选): 特征缓存，默认为 None。
            feat_idx (list of int, 可选): 特征索引列表，默认为 [0]。

        返回:
            torch.Tensor: 输出张量，经过解码器处理后的重构数据。
        """
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            # 缓存最后 CACHE_T 个时间步的数据
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # 如果缓存长度小于 2 且缓存不为 None，则将缓存与最后一张帧连接起来
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            # 应用第一个因果卷积层，传入缓存
            x = self.conv1(x, feat_cache[idx])
            # 更新缓存
            feat_cache[idx] = cache_x
            # 更新索引
            feat_idx[0] += 1
        else:
            # 否则，正常应用第一个因果卷积层
            x = self.conv1(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                # 应用残差块，传入缓存
                x = layer(x, feat_cache, feat_idx)
            else:
                # 否则，正常应用层
                x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                # 应用上采样层，传入缓存
                x = layer(x, feat_cache, feat_idx)
            else:
                # 否则，正常应用上采样层
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                # 缓存最后 CACHE_T 个时间步的数据
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # 如果缓存长度小于 2 且缓存不为 None，则将缓存与最后一张帧连接起来
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                # 应用因果卷积层，传入缓存
                x = layer(x, feat_cache[idx])
                # 更新缓存
                feat_cache[idx] = cache_x
                # 更新索引
                feat_idx[0] += 1
            else:
                # 否则，正常应用层
                x = layer(x)
        return x


def count_conv3d(model):
    """
    计算模型中 CausalConv3d 卷积层的数量。

    参数:
        model (nn.Module): 要计算的模型。

    返回:
        int: CausalConv3d 卷积层的数量。
    """
    # 初始化计数器
    count = 0
    for m in model.modules():
        # 如果当前模块是 CausalConv3d，则计数器加一
        if isinstance(m, CausalConv3d):
            count += 1
    # 返回计数结果
    return count


class WanVAE(nn.Module):
    """
    WanVAE 类，实现了一个变分自编码器（VAE）。
    该 VAE 使用 3D 卷积进行编码和解码，并结合了因果卷积和自注意力机制。
    """
    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        """
        初始化 WanVAE 模块。

        参数:
            dim (int, 可选): 基础维度，默认为 128。
            z_dim (int, 可选): 潜在空间的维度，默认为 4。
            dim_mult (list of int, 可选): 每个块的维度倍数列表，默认为 [1, 2, 4, 4]。
            num_res_blocks (int, 可选): 每个块的残差块数量，默认为 2。
            attn_scales (list of float, 可选): 自注意力机制的应用尺度列表，默认为空列表。
            temperal_downsample (list of bool, 可选): 是否进行时间步下采样，默认为 [True, True, False]。
            dropout (float, 可选): Dropout 概率，默认为 0.0。
        """
        super().__init__()
        self.dim = dim  # 设置基础维度
        self.z_dim = z_dim  # 设置潜在空间维度
        self.dim_mult = dim_mult  # 设置维度倍数列表
        self.num_res_blocks = num_res_blocks  # 设置残差块数量
        self.attn_scales = attn_scales  # 设置自注意力尺度列表
        self.temperal_downsample = temperal_downsample  # 设置时间步下采样标志列表
        self.temperal_upsample = temperal_downsample[::-1]  # 设置时间步上采样标志列表

        # 定义编码器和解码器模块
        # 初始化编码器
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout)
        # 初始化 1x1 卷积层
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        # 初始化 1x1 卷积层
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        # 初始化解码器
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout)

    def forward(self, x):
        """
        前向传播函数，应用 VAE。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            tuple: 包含重构数据、均值和方差的对数。
        """
        # 编码输入数据，得到均值和方差的对数
        mu, log_var = self.encode(x)
        # 重参数化，得到潜在变量
        z = self.reparameterize(mu, log_var)
        # 解码潜在变量，得到重构数据
        x_recon = self.decode(z)
        # 返回重构数据、均值和方差的对数
        return x_recon, mu, log_var

    def encode(self, x):
        """
        编码函数，对输入数据进行编码。

        参数:
            x (torch.Tensor): 输入数据，形状为 (batch_size, channels, time_steps, height, width)。

        返回:
            tuple: 包含均值和方差的对数。
        """
        # 清除缓存
        self.clear_cache()
        # 获取时间步长度
        t = x.shape[2]
        # 计算迭代次数
        iter_ = 1 + (t - 1) // 4
        ## 对encode输入的x，按时间拆分为1、4、4、4....
        for i in range(iter_):
            # 重置编码器卷积索引
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)  # 处理第一个时间步
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)  # 处理后续时间步
                # 连接输出
                out = torch.cat([out, out_], 2)
        # 分割输出为均值和方差的对数
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        # 清除缓存
        self.clear_cache()
        # 返回均值和方差的对数
        return mu

    def decode(self, z):
        """
        解码函数，对潜在变量进行解码。

        参数:
            z (torch.Tensor): 潜在变量，形状为 (batch_size, z_dim, time_steps, height, width)。

        返回:
            torch.Tensor: 重构数据。
        """
        # 清除缓存
        self.clear_cache()
        # z: [b,c,t,h,w]
        
        # 获取时间步长度
        iter_ = z.shape[2]
        # 应用 1x1 卷积层
        x = self.conv2(z)
        for i in range(iter_):
            # 重置解码器卷积索引
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)  # 处理第一个时间步
            else:
                out_ = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)  # 处理后续时间步
                out = torch.cat([out, out_], 2)  # 连接输出
        # 清除缓存
        self.clear_cache()
        # 返回重构数据
        return out

    def reparameterize(self, mu, log_var):
        """
        重参数化函数，用于生成潜在变量。

        参数:
            mu (torch.Tensor): 均值。
            log_var (torch.Tensor): 方差的对数。

        返回:
            torch.Tensor: 潜在变量。
        """
        std = torch.exp(0.5 * log_var)  # 计算标准差
        eps = torch.randn_like(std)  # 从标准正态分布中采样
        # 返回潜在变量
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        """
        采样函数，用于生成样本。

        参数:
            imgs (torch.Tensor): 输入图像，形状为 (batch_size, channels, time_steps, height, width)。
            deterministic (bool, 可选): 是否使用确定性采样，默认为 False。

        返回:
            torch.Tensor: 采样后的潜在变量。
        """
        # 编码输入图像，得到均值和方差的对数
        mu, log_var = self.encode(imgs)
        if deterministic:
            # 如果是确定性采样，则返回均值
            return mu
        # 计算标准差，并限制范围
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        # 返回采样后的潜在变量
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        """
        清除缓存。
        """
        # 计算解码器中 CausalConv3d 卷积层的数量
        self._conv_num = count_conv3d(self.decoder)
        # 重置卷积索引
        self._conv_idx = [0]
        # 初始化特征缓存列表
        self._feat_map = [None] * self._conv_num
        
        # 计算编码器中 CausalConv3d 卷积层的数量
        self._enc_conv_num = count_conv3d(self.encoder)
        # 重置编码器卷积索引
        self._enc_conv_idx = [0]
        # 初始化编码器特征缓存列表
        self._enc_feat_map = [None] * self._enc_conv_num
