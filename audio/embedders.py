# code adapted from: https://github.com/Stability-AI/stable-audio-tools

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Union
from einops import rearrange
import math
import ops


class LearnedPositionalEmbedding(nn.Module):
    """
    LearnedPositionalEmbedding 类，用于连续时间的位置编码。
    该类通过学习的位置嵌入为输入张量添加位置信息，适用于连续时间数据。
    """

    def __init__(self, dim: int):
        """
        初始化 LearnedPositionalEmbedding 模块。

        参数:
            dim (int): 嵌入的维度，必须为偶数。
        """
        super().__init__()

        # 确保维度为偶数
        assert (dim % 2) == 0

        # 计算维度的一半
        half_dim = dim // 2

        # 初始化可学习的权重参数，形状为 (half_dim,)
        self.weights = nn.Parameter(torch.empty(half_dim))

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播函数，生成位置嵌入。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size,)。

        返回:
            torch.Tensor: 位置嵌入张量，形状为 (batch_size, dim + 1)。
        """
        # 将输入张量重塑为 (batch_size, 1)
        x = rearrange(x, "b -> b 1")

        # 计算频率，形状为 (1, half_dim)
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi

        # 计算正弦和余弦，形状为 (batch_size, half_dim * 2)
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)

        # 将输入张量和频率张量连接起来，形状为 (batch_size, dim + 1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


def TimePositionalEmbedding(dim: int, out_features: int) -> nn.Module:
    """
    TimePositionalEmbedding 函数，用于生成时间位置嵌入。

    参数:
        dim (int): 输入特征的维度。
        out_features (int): 输出特征的维度。

    返回:
        nn.Module: 包含 LearnedPositionalEmbedding 和线性变换的序列模块。
    """
    return nn.Sequential(
        LearnedPositionalEmbedding(dim),  # 第一个模块：学习的位置嵌入
        ops.manual_cast.Linear(in_features=dim + 1, out_features=out_features), # 第二个模块：线性变换
    )


class NumberEmbedder(nn.Module):
    """
    NumberEmbedder 类，用于将数值转换为嵌入。
    该类通过 TimePositionalEmbedding 将数值转换为嵌入，并重塑为所需的形状。
    """
    def __init__(
        self,
        features: int,
        dim: int = 256,
    ):
        """
        初始化 NumberEmbedder 模块。

        参数:
            features (int): 输出特征的维度。
            dim (int, 可选): 嵌入的维度，默认为 256。
        """
        super().__init__()
        # 设置输出特征的维度
        self.features = features
        # 初始化时间位置嵌入模块
        self.embedding = TimePositionalEmbedding(dim=dim, out_features=features)

    def forward(self, x: Union[List[float], Tensor]) -> Tensor:
        """
        前向传播函数，将输入数值转换为嵌入。

        参数:
            x (list of float 或 torch.Tensor): 输入的数值列表或张量。

        返回:
            torch.Tensor: 嵌入后的张量，形状为 (..., features)。
        """
        if not torch.is_tensor(x):
            # 如果输入不是张量，则将其转换为张量，并移动到与嵌入器相同的设备
            device = next(self.embedding.parameters()).device
            x = torch.tensor(x, device=device)
        # 确保输入是张量
        assert isinstance(x, Tensor)
        # 保存原始形状
        shape = x.shape
        # 重塑张量以适应嵌入器
        x = rearrange(x, "... -> (...)")
        # 应用嵌入器
        embedding = self.embedding(x)
        # 将嵌入后的张量重塑回原始形状，并添加特征维度
        x = embedding.view(*shape, self.features)
        # 返回嵌入后的张量
        return x  


class Conditioner(nn.Module):
    """
    Conditioner 基类，用于条件嵌入。
    该类定义了条件器的基类，具体的实现由子类完成。
    条件器的主要功能是将输入数据转换为嵌入向量，并根据需要调整输出维度。
    """
    def __init__(
            self,
            dim: int,
            output_dim: int,
            project_out: bool = False
            ):
        """
        初始化 Conditioner 基类。

        参数:
            dim (int): 输入特征的维度。
            output_dim (int): 输出特征的维度。
            project_out (bool, 可选): 是否进行线性投影以调整输出维度，默认为 False。
        """

        super().__init__()

        # 设置输入特征的维度
        self.dim = dim
        # 设置输出特征的维度
        self.output_dim = output_dim
        # 如果输入和输出维度不同，或者需要投影，则初始化线性投影层；否则，使用恒等函数
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()

    def forward(self, x):
        """
        前向传播函数，需要在子类中实现。
        """
        raise NotImplementedError()


class NumberConditioner(Conditioner):
    """
    NumberConditioner 类，继承自 Conditioner，用于数值条件嵌入。
    该类将数值列表归一化，并生成对应的嵌入列表。
    """
    def __init__(self,
                output_dim: int,
                min_val: float=0,
                max_val: float=1
                ):
        """
        初始化 NumberConditioner。

        参数:
            output_dim (int): 输出特征的维度。
            min_val (float, 可选): 数值范围的下限，默认为 0。
            max_val (float, 可选): 数值范围的上限，默认为 1。
        """
        super().__init__(output_dim, output_dim)

        # 设置数值范围的下限
        self.min_val = min_val
        # 设置数值范围的上限
        self.max_val = max_val

        # 初始化数值嵌入器，输出维度为 output_dim
        self.embedder = NumberEmbedder(features=output_dim)

    def forward(self, floats, device=None):
        """
        前向传播函数，将数值列表转换为嵌入。

        参数:
            floats (list): 输入的数值列表。
            device (torch.device, 可选): 设备，默认为 None。

        返回:
            list: 包含嵌入张量和全 1 张量的列表。
                  - float_embeds (torch.Tensor): 嵌入张量，形状为 (batch_size, 1, output_dim)。
                  - ones (torch.Tensor): 全 1 张量，形状为 (batch_size, 1)。
        """
        # 将输入的数值列表转换为浮点数列表
        floats = [float(x) for x in floats]

        if device is None:
            # 如果未指定设备，则获取嵌入器参数所在的设备
            device = next(self.embedder.parameters()).device

        # 将数值列表转换为张量，并移动到指定设备
        floats = torch.tensor(floats).to(device)

        # 将数值限制在 [min_val, max_val] 范围内
        floats = floats.clamp(self.min_val, self.max_val)

        # 归一化数值到 [0, 1] 范围
        normalized_floats = (floats - self.min_val) / (self.max_val - self.min_val)

        # 获取嵌入器的参数数据类型
        embedder_dtype = next(self.embedder.parameters()).dtype
        # 将归一化后的数值转换为与嵌入器相同的数据类型
        normalized_floats = normalized_floats.to(embedder_dtype)

        # 应用数值嵌入器，生成嵌入张量，形状为 (batch_size, output_dim)
        # 在第二维添加一个维度，形状变为 (batch_size, 1, output_dim)
        float_embeds = self.embedder(normalized_floats).unsqueeze(1)

        # 生成一个全 1 张量，形状为 (batch_size, 1)
        # 返回包含嵌入张量和全 1 张量的列表
        return [float_embeds, torch.ones(float_embeds.shape[0], 1).to(device)]
