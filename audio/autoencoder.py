# code adapted from: https://github.com/Stability-AI/stable-audio-tools

import torch
from torch import nn
from typing import Literal
import math
import ops
ops = ops.disable_weight_init


def vae_sample(mean, scale):
    """
    从给定的均值和尺度参数中采样，并计算 KL 散度。

    参数:
        mean (torch.Tensor): 均值张量，形状为 (batch_size, latent_dim)。
        scale (torch.Tensor): 尺度参数张量，形状为 (batch_size, latent_dim)。

    返回:
        tuple: 包含采样后的潜在变量张量和 KL 散度。
               - latents (torch.Tensor): 采样后的潜在变量，形状与 `mean` 相同。
               - kl (torch.Tensor): 计算得到的 KL 散度标量。
    """
    # 计算标准差，使用 softplus 函数确保标准差为正数，并加上一个小的常数以避免数值不稳定    
    stdev = nn.functional.softplus(scale) + 1e-4
    # 计算方差
    var = stdev * stdev
    # 计算对数方差
    logvar = torch.log(var)
    # 从标准正态分布中采样，并与均值和标准差结合，生成潜在变量
    latents = torch.randn_like(mean) * stdev + mean

    # 计算 KL 散度项：
    # KL = (mean^2 + var - logvar - 1).sum(1).mean()
    # 其中，sum(1) 对每个样本的维度求和，mean() 对批次维度求平均
    kl = (mean * mean + var - logvar - 1).sum(1).mean()

    return latents, kl


class VAEBottleneck(nn.Module):
    """
    VAE 瓶颈模块，用于编码和解码输入数据。
    该模块实现了变分自编码器（VAE）的基本功能，包括编码和解码过程。
    """
    def __init__(self):
        super().__init__()
        # 标记是否为离散 VAE，默认为 False
        self.is_discrete = False

    def encode(self, x, return_info=False, **kwargs):
        """
        编码输入数据，生成潜在变量及其 KL 散度。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, channels, height, width)。
            return_info (bool, 可选): 是否返回附加信息，默认为 False。

        返回:
            tuple: 如果 `return_info` 为 True，则返回 (latent, info)；
                   否则，仅返回 latent。
                   - latent (torch.Tensor): 编码后的潜在变量。
                   - info (dict, 可选): 包含 KL 散度等附加信息的字典。
        """
        # 初始化附加信息字典
        info = {}

        # 将输入张量沿通道维度拆分为均值和尺度参数
        mean, scale = x.chunk(2, dim=1)

        # 从均值和尺度参数中采样，并计算 KL 散度
        x, kl = vae_sample(mean, scale)

        # 将 KL 散度存储在 info 字典中
        info["kl"] = kl

        if return_info:
            # 如果需要返回附加信息，则返回潜在变量和 info
            return x, info
        else:
            # 否则，仅返回潜在变量
            return x

    def decode(self, x):
        """
        解码潜在变量，生成重构数据。

        参数:
            x (torch.Tensor): 潜在变量张量。

        返回:
            torch.Tensor: 重构后的输出张量。
        """
        # 简单的解码过程，直接返回输入
        return x


def snake_beta(x, alpha, beta):
    """
    SnakeBeta 激活函数。

    参数:
        x (torch.Tensor): 输入张量。
        alpha (float): 控制频率的参数。
        beta (float): 控制幅度的参数。

    返回:
        torch.Tensor: 应用了 SnakeBeta 激活函数后的张量。
    """
    return x + (1.0 / (beta + 0.000000001)) * pow(torch.sin(x * alpha), 2)


# Adapted from https://github.com/NVIDIA/BigVGAN/blob/main/activations.py under MIT license
class SnakeBeta(nn.Module):
    """
    SnakeBeta 激活函数模块。
    该模块实现了 SnakeBeta 激活函数，可以通过可训练参数进行调节。
    """

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        """
        初始化 SnakeBeta 模块。

        参数:
            in_features (int): 输入特征的维度。
            alpha (float, 可选): 控制频率的初始参数，默认为 1.0。
            alpha_trainable (bool, 可选): 是否训练 alpha 参数，默认为 True。
            alpha_logscale (bool, 可选): 是否使用对数尺度表示 alpha，默认为 True。
        """
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # 设置 alpha 的表示方式
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale: # log scale alphas initialized to zeros
            # 如果使用对数尺度，则初始化 alpha 为零
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else: # linear scale alphas initialized to ones
            # 否则，初始化 alpha 为 1
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)

        # self.alpha.requires_grad = alpha_trainable
        # self.beta.requires_grad = alpha_trainable

        # 防止除零错误的小常数
        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        前向传播函数，应用 SnakeBeta 激活函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features, ...)。

        返回:
            torch.Tensor: 应用了 SnakeBeta 后的输出张量。
        """
        # 将 alpha 和 beta 参数调整为与输入张量相同的设备，并扩展维度以匹配输入形状
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1).to(x.device) # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1).to(x.device)
        if self.alpha_logscale:
            # 如果使用对数尺度，则将 alpha 和 beta 转换为指数尺度
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        # 应用 SnakeBeta 激活函数
        x = snake_beta(x, alpha, beta)

        return x


def WNConv1d(*args, **kwargs):
    """
    应用权重归一化的 1D 卷积层。

    参数:
        *args: 传递给 Conv1d 的位置参数。
        **kwargs: 传递给 Conv1d 的关键字参数。

    返回:
        nn.Module: 应用了权重归一化的 Conv1d 模块。
    """
    return torch.nn.utils.parametrizations.weight_norm(ops.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    """
    应用权重归一化的 1D 转置卷积层。

    参数:
        *args: 传递给 ConvTranspose1d 的位置参数。
        **kwargs: 传递给 ConvTranspose1d 的关键字参数。

    返回:
        nn.Module: 应用了权重归一化的 ConvTranspose1d 模块。
    """
    return torch.nn.utils.parametrizations.weight_norm(ops.ConvTranspose1d(*args, **kwargs))


def get_activation(activation: Literal["elu", "snake", "none"], antialias=False, channels=None) -> nn.Module:
    """
    获取指定的激活函数模块。

    参数:
        activation (str): 激活函数的名称，可以是 "elu", "snake", "none"。
        antialias (bool, 可选): 是否应用抗锯齿处理，默认为 False。
        channels (int, 可选): 通道数，用于 SnakeBeta 激活函数，默认为 None。

    返回:
        nn.Module: 激活函数模块。

    异常:
        ValueError: 如果激活函数名称未知，则引发 ValueError。
    """
    if activation == "elu":
        act = torch.nn.ELU()
    elif activation == "snake":
        act = SnakeBeta(channels)
    elif activation == "none":
        act = torch.nn.Identity()
    else:
        raise ValueError(f"Unknown activation {activation}")

    if antialias:
        # 应用抗锯齿处理
        act = Activation1d(act)  # noqa: F821 Activation1d is not defined

    return act


class ResidualUnit(nn.Module):
    """
    残差单元（ResidualUnit）类。
    该类实现了带有残差连接的基础单元，包含激活函数和卷积层。
    """
    def __init__(self, in_channels, out_channels, dilation, use_snake=False, antialias_activation=False):
        """
        初始化残差单元。

        参数:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            dilation (int): 卷积层的膨胀率，用于控制感受野。
            use_snake (bool, 可选): 是否使用 Snake 激活函数，默认为 False。
            antialias_activation (bool, 可选): 是否对激活函数应用抗锯齿处理，默认为 False。
        """
        super().__init__()

        # 设置膨胀率
        self.dilation = dilation

        # 计算填充大小，以保持输出尺寸不变
        padding = (dilation * (7-1)) // 2

        # 定义残差单元的层序列
        self.layers = nn.Sequential(
            # 第一个激活函数
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            # 第一个权重归一化的 1D 卷积层
            WNConv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=7, dilation=dilation, padding=padding),
            # 第二个激活函数
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            # 第二个权重归一化的 1D 卷积层
            WNConv1d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1)
        )

    def forward(self, x):
        """
        前向传播函数，应用残差单元。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 输出张量，经过残差单元处理后的结果。
        """
        # 保存输入张量作为残差
        res = x

        # 使用残差单元的层序列处理输入张量
        x = self.layers(x)

        # 将处理后的张量与残差相加
        return x + res


class EncoderBlock(nn.Module):
    """
    编码器块（EncoderBlock）类。
    该类实现了编码器的基础块，包含多个残差单元和一个下采样卷积层。
    """
    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False):
        """
        初始化编码器块。

        参数:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            stride (int): 卷积层的步幅，用于下采样。
            use_snake (bool, 可选): 是否使用 Snake 激活函数，默认为 False。
            antialias_activation (bool, 可选): 是否对激活函数应用抗锯齿处理，默认为 False。
        """
        super().__init__()

        # 定义编码器块的层序列
        self.layers = nn.Sequential(
            ResidualUnit(in_channels=in_channels,  # 第一个残差单元
                         out_channels=in_channels, dilation=1, use_snake=use_snake),
            ResidualUnit(in_channels=in_channels,  # 第二个残差单元
                         out_channels=in_channels, dilation=3, use_snake=use_snake),
            ResidualUnit(in_channels=in_channels,  # 第三个残差单元
                         out_channels=in_channels, dilation=9, use_snake=use_snake),
            # 激活函数
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            # 下采样卷积层
            WNConv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2)),
        )

    def forward(self, x):
        """
        前向传播函数，应用编码器块。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 输出张量，经过编码器块处理后的结果。
        """
        # 应用层序列并返回结果
        return self.layers(x)


class DecoderBlock(nn.Module):
    """
    解码器块（DecoderBlock）类。
    该类实现了解码器的基础块，包含一个上采样层和多个残差单元。
    """
    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False, use_nearest_upsample=False):
        """
        初始化解码器块。

        参数:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            stride (int): 上采样层的步幅，用于上采样。
            use_snake (bool, 可选): 是否使用 Snake 激活函数，默认为 False。
            antialias_activation (bool, 可选): 是否对激活函数应用抗锯齿处理，默认为 False。
            use_nearest_upsample (bool, 可选): 是否使用最近邻上采样，默认为 False。
        """
        super().__init__()

        # 根据是否使用最近邻上采样，选择上采样层
        if use_nearest_upsample:
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="nearest"),  # 最近邻上采样
                WNConv1d(in_channels=in_channels,  # 权重归一化的 1D 卷积层
                        out_channels=out_channels,
                        kernel_size=2*stride,
                        stride=1,
                        bias=False,
                        padding='same')
            )
        else:
            # 权重归一化的 1D 转置卷积层
            upsample_layer = WNConvTranspose1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2))

        # 定义解码器块的层序列
        self.layers = nn.Sequential(
            # 激活函数
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            upsample_layer,  # 上采样层
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,  # 第一个残差单元
                         dilation=1, use_snake=use_snake),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,  # 第二个残差单元
                         dilation=3, use_snake=use_snake),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,  # 第三个残差单元
                         dilation=9, use_snake=use_snake),
        )

    def forward(self, x):
        """
        前向传播函数，应用解码器块。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 输出张量，经过解码器块处理后的结果。
        """
        # 应用层序列并返回结果
        return self.layers(x)


class OobleckEncoder(nn.Module):
    """
    Oobleck 编码器类。
    该类实现了 Oobleck 模型的编码器部分，包含多个编码器块和卷积层。
    """
    def __init__(self,
                 in_channels=2,
                 channels=128,
                 latent_dim=32,
                 c_mults = [1, 2, 4, 8],
                 strides = [2, 4, 8, 8],
                 use_snake=False,
                 antialias_activation=False
        ):
        """
        初始化 Oobleck 编码器。

        参数:
            in_channels (int, 可选): 输入通道数，默认为 2。
            channels (int, 可选): 基本通道数，默认为 128。
            latent_dim (int, 可选): 潜在空间的维度，默认为 32。
            c_mults (list, 可选): 每个编码器块的通道倍数列表，默认为 [1, 2, 4, 8]。
            strides (list, 可选): 每个编码器块的下采样步幅列表，默认为 [2, 4, 8, 8]。
            use_snake (bool, 可选): 是否使用 Snake 激活函数，默认为 False。
            antialias_activation (bool, 可选): 是否对激活函数应用抗锯齿处理，默认为 False。
        """
        super().__init__()

        # 在 c_mults 列表前添加 1，以匹配输入通道数
        c_mults = [1] + c_mults

        # 计算编码器块的深度
        self.depth = len(c_mults)

        # 第一个卷积层
        layers = [
            WNConv1d(in_channels=in_channels, out_channels=c_mults[0] * channels, kernel_size=7, padding=3)
        ]

        for i in range(self.depth-1):
            # 添加编码器块
            layers += [EncoderBlock(in_channels=c_mults[i]*channels, out_channels=c_mults[i+1]*channels, stride=strides[i], use_snake=use_snake)]

        layers += [
            # 激活函数
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[-1] * channels),
            # 最后一个卷积层，输出潜在空间
            WNConv1d(in_channels=c_mults[-1]*channels, out_channels=latent_dim, kernel_size=3, padding=1)
        ]

        # 将所有层组合成一个序列
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播函数，应用编码器。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 输出张量，经过编码器处理后的潜在空间。
        """
        # 应用层序列并返回结果
        return self.layers(x)


class OobleckDecoder(nn.Module):
    """
    Oobleck 解码器类。
    该类实现了 Oobleck 模型的解码器部分，包含多个解码器块和卷积层。
    """
    def __init__(self,
                 out_channels=2,
                 channels=128,
                 latent_dim=32,
                 c_mults = [1, 2, 4, 8],
                 strides = [2, 4, 8, 8],
                 use_snake=False,
                 antialias_activation=False,
                 use_nearest_upsample=False,
                 final_tanh=True):
        """
        初始化 Oobleck 解码器。

        参数:
            out_channels (int, 可选): 输出通道数，默认为 2。
            channels (int, 可选): 基本通道数，默认为 128。
            latent_dim (int, 可选): 潜在空间的维度，默认为 32。
            c_mults (list, 可选): 每个解码器块的通道倍数列表，默认为 [1, 2, 4, 8]。
            strides (list, 可选): 每个解码器块的上采样步幅列表，默认为 [2, 4, 8, 8]。
            use_snake (bool, 可选): 是否使用 Snake 激活函数，默认为 False。
            antialias_activation (bool, 可选): 是否对激活函数应用抗锯齿处理，默认为 False。
            use_nearest_upsample (bool, 可选): 是否使用最近邻上采样，默认为 False。
            final_tanh (bool, 可选): 是否在最后使用 Tanh 激活函数，默认为 True。
        """
        super().__init__()

        # 在 c_mults 列表前添加 1，以匹配输出通道数
        c_mults = [1] + c_mults

        # 计算解码器块的深度
        self.depth = len(c_mults)

        # 第一个卷积层，输入潜在空间
        layers = [
            WNConv1d(in_channels=latent_dim, out_channels=c_mults[-1]*channels, kernel_size=7, padding=3),
        ]

        for i in range(self.depth-1, 0, -1):
            # 添加解码器块
            layers += [DecoderBlock(
                in_channels=c_mults[i]*channels,
                out_channels=c_mults[i-1]*channels,
                stride=strides[i-1],
                use_snake=use_snake,
                antialias_activation=antialias_activation,
                use_nearest_upsample=use_nearest_upsample
                )
            ]

        layers += [
            # 激活函数
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[0] * channels),
            # 最后一个卷积层，输出最终结果
            WNConv1d(in_channels=c_mults[0] * channels, out_channels=out_channels, kernel_size=7, padding=3, bias=False),
            # 最后使用 Tanh 激活函数
            nn.Tanh() if final_tanh else nn.Identity()
        ]

        # 将所有层组合成一个序列
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播函数，应用解码器。

        参数:
            x (torch.Tensor): 输入潜在空间张量。

        返回:
            torch.Tensor: 输出张量，经过解码器处理后的重构数据。
        """
        # 应用层序列并返回结果
        return self.layers(x)


class AudioOobleckVAE(nn.Module):
    """
    AudioOobleckVAE 类，实现了音频 Oobleck 模型的变分自编码器（VAE）版本。
    该模型结合了编码器、解码器和 VAE 瓶颈模块，用于音频数据的处理和重构。
    """
    def __init__(self,
                 in_channels=2,
                 channels=128,
                 latent_dim=64,
                 c_mults = [1, 2, 4, 8, 16],
                 strides = [2, 4, 4, 8, 8],
                 use_snake=True,
                 antialias_activation=False,
                 use_nearest_upsample=False,
                 final_tanh=False):
        """
        初始化 AudioOobleckVAE。

        参数:
            in_channels (int, 可选): 输入通道数，默认为 2。
            channels (int, 可选): 基本通道数，默认为 128。
            latent_dim (int, 可选): 潜在空间的维度，默认为 64。
            c_mults (list, 可选): 每个编码器和解码器块的通道倍数列表，默认为 [1, 2, 4, 8, 16]。
            strides (list, 可选): 每个编码器和解码器块的下采样和上采样步幅列表，默认为 [2, 4, 4, 8, 8]。
            use_snake (bool, 可选): 是否使用 Snake 激活函数，默认为 True。
            antialias_activation (bool, 可选): 是否对激活函数应用抗锯齿处理，默认为 False。
            use_nearest_upsample (bool, 可选): 是否使用最近邻上采样，默认为 False。
            final_tanh (bool, 可选): 是否在最后使用 Tanh 激活函数，默认为 False。
        """
        super().__init__()
        # 初始化编码器
        self.encoder = OobleckEncoder(in_channels, channels, latent_dim * 2, c_mults, strides, use_snake, antialias_activation)
        # 初始化解码器
        self.decoder = OobleckDecoder(in_channels, channels, latent_dim, c_mults, strides, use_snake, antialias_activation,
                                      use_nearest_upsample=use_nearest_upsample, final_tanh=final_tanh)
        # 初始化 VAE 瓶颈模块
        self.bottleneck = VAEBottleneck()

    def encode(self, x):
        """
        编码输入数据，生成潜在变量及其 KL 散度。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            tuple: 包含潜在变量和附加信息的元组。
                   - latent (torch.Tensor): 编码后的潜在变量。
                   - info (dict): 包含 KL 散度等附加信息的字典。
        """
        # 通过编码器处理输入，并传递给瓶颈模块进行编码
        return self.bottleneck.encode(self.encoder(x))

    def decode(self, x):
        """
        解码潜在变量，重构数据。

        参数:
            x (torch.Tensor): 潜在变量张量。

        返回:
            torch.Tensor: 重构后的输出张量。
        """
        # 通过瓶颈模块解码，并传递给解码器进行重构
        return self.decoder(self.bottleneck.decode(x))

