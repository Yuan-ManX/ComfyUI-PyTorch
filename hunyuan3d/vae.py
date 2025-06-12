# Original: https://github.com/Tencent/Hunyuan3D-2/blob/main/hy3dgen/shapegen/models/autoencoders/model.py
# Since the header on their VAE source file was a bit confusing we asked for permission to use this code from tencent under the GPL license used in ComfyUI.

import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import Union, Tuple, List, Callable, Optional

import numpy as np
from einops import repeat, rearrange
from tqdm import tqdm
import logging

import ops
ops = ops.disable_weight_init


def generate_dense_grid_points(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    octree_resolution: int,
    indexing: str = "ij",
):
    """
    生成一个密集的网格点，用于3D体积解码。

    参数:
        bbox_min (np.ndarray): 边界框的最小坐标，形状为 (3,)。
        bbox_max (np.ndarray): 边界框的最大坐标，形状为 (3,)。
        octree_resolution (int): 八叉树分辨率，决定了网格的细粒度。
        indexing (str): 网格索引方式，默认为 "ij"。可选值包括 "xy" 和其他。

    返回:
        Tuple[np.ndarray, List[int], np.ndarray]:
            - xyz (np.ndarray): 生成的网格点，形状为 (N, 3)。
            - grid_size (List[int]): 网格的尺寸，列表包含三个维度的大小。
            - length (np.ndarray): 边界框在每个维度上的长度，形状为 (3,)。
    """
    # 计算边界框在每个维度上的长度
    length = bbox_max - bbox_min

    # 获取八叉树分辨率，确保为整数
    num_cells = octree_resolution  # 每个维度上的单元格数量
    
    # 在每个维度上生成均匀分布的坐标点
    # 从 bbox_min[0] 到 bbox_max[0]，生成 num_cells + 1 个点
    x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
    # 从 bbox_min[1] 到 bbox_max[1]，生成 num_cells + 1 个点
    y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
    # 从 bbox_min[2] 到 bbox_max[2]，生成 num_cells + 1 个点
    z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)

    # 使用 np.meshgrid 生成网格坐标
    # indexing 参数决定了输出网格的索引方式
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)

    # 将三个坐标轴的网格堆叠在一起，形成一个 (N, N, N, 3) 的张量
    xyz = np.stack((xs, ys, zs), axis=-1)

    # 定义网格的尺寸
    # 每个维度上的网格点数量
    grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]

    # 返回网格点、网格尺寸和边界框长度
    return xyz, grid_size, length


class VanillaVolumeDecoder:
    """
    一个基础的体积解码器，用于将潜在向量解码为3D体积数据。
    """
    # 禁用梯度计算，节省内存并加快推理速度
    @torch.no_grad()
    def __call__(
        self,
        latents: torch.FloatTensor,
        geo_decoder: Callable,
        bounds: Union[Tuple[float], List[float], float] = 1.01,
        num_chunks: int = 10000,
        octree_resolution: int = None,
        enable_pbar: bool = True,
        **kwargs,
    ):
        """
        前向调用方法，用于将潜在向量解码为3D体积。

        参数:
            latents (torch.FloatTensor): 输入的潜在向量，形状为 (batch_size, latent_dim)。
            geo_decoder (Callable): 用于解码查询点的解码器函数。
            bounds (Union[Tuple[float], List[float], float]): 边界框范围，可以是单个浮点数或包含六个元素的列表/元组。
            num_chunks (int): 每个批次的查询点数量，默认为10000。
            octree_resolution (int): 八叉树分辨率，用于生成网格点。
            enable_pbar (bool): 是否启用进度条，默认为True。
            **kwargs: 其他可选的关键字参数。

        返回:
            torch.FloatTensor: 解码后的3D体积数据，形状为 (batch_size, N, N, N)。
        """
        # 获取设备（CPU或GPU）
        device = latents.device
        # 获取数据类型
        dtype = latents.dtype
        # 获取批大小
        batch_size = latents.shape[0]

        # 1. 生成查询点
        if isinstance(bounds, float):
            # 如果 bounds 是一个浮点数，则将其转换为包含六个元素的列表，假设边界框是对称的
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        # 分离最小和最大边界
        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        xyz_samples, grid_size, length = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_resolution=octree_resolution,
            indexing="ij"
        )  # 生成密集的网格点

        # 将生成的网格点转换为 PyTorch 张量，并移动到目标设备
        xyz_samples = torch.from_numpy(xyz_samples).to(device, dtype=dtype).contiguous().reshape(-1, 3)

        # 2. 将潜在向量解码为3D体积
        batch_logits = []  # 用于存储每个批次的解码结果
        for start in tqdm(range(0, xyz_samples.shape[0], num_chunks), desc="Volume Decoding",
                          disable=not enable_pbar):
            # 获取当前批次的查询点
            chunk_queries = xyz_samples[start: start + num_chunks, :]
            # 重复查询点以匹配批大小
            chunk_queries = repeat(chunk_queries, "p c -> b p c", b=batch_size)
            # 使用解码器解码查询点
            logits = geo_decoder(queries=chunk_queries, latents=latents)
            # 将解码结果添加到列表中
            batch_logits.append(logits)

        # 将所有批次的解码结果连接起来
        grid_logits = torch.cat(batch_logits, dim=1)
        # 重塑为3D体积的形状
        grid_logits = grid_logits.view((batch_size, *grid_size)).float()

        # 返回解码后的3D体积数据
        return grid_logits


class FourierEmbedder(nn.Module):
    """
    Fourier 位置编码器。该编码器将输入张量 `x` 的每个特征维度 `x[..., i]` 转换为包含正弦和余弦函数的多频域表示。
    具体来说，对于每个特征维度 `i`，输出包含：
        [
            sin(x[..., i]),
            sin(f_1 * x[..., i]),
            sin(f_2 * x[..., i]),
            ...
            sin(f_N * x[..., i]),
            cos(x[..., i]),
            cos(f_1 * x[..., i]),
            cos(f_2 * x[..., i]),
            ...
            cos(f_N * x[..., i]),
            x[..., i]     # 仅当 include_input 为 True 时存在。
        ]
    其中，f_i 是频率。

    频率空间定义为 [0 / num_freqs, 1 / num_freqs, 2 / num_freqs, ..., (num_freqs - 1) / num_freqs]。
    如果 `logspace` 为 True，则频率 f_i 为 [2^(0 / num_freqs), ..., 2^(i / num_freqs), ...]；
    否则，频率在 [1.0, 2^(num_freqs - 1)] 之间线性分布。

    参数:
        num_freqs (int): 频率的数量，默认为6。
        logspace (bool): 如果为 True，则频率为对数分布；否则，频率为线性分布。默认为 True。
        input_dim (int): 输入维度，默认为3。
        include_input (bool): 是否包含原始输入张量，默认为 True。
        include_pi (bool): 是否在频率中包含 π，默认为 True。

    属性:
        frequencies (torch.Tensor): 频率张量。如果 `logspace` 为 True，则频率为 [2^(0 / num_freqs), ..., 2^(i / num_freqs), ...]；
                                     否则，频率在 [1.0, 2^(num_freqs - 1)] 之间线性分布。
        out_dim (int): 编码后的输出维度。如果 `include_input` 为 True，则为 `input_dim * (num_freqs * 2 + 1)`；
                      否则，为 `input_dim * num_freqs * 2`。
    """

    def __init__(self,
                 num_freqs: int = 6,
                 logspace: bool = True,
                 input_dim: int = 3,
                 include_input: bool = True,
                 include_pi: bool = True) -> None:

        """
        初始化 FourierEmbedder。

        参数:
            num_freqs (int): 频率的数量，默认为6。
            logspace (bool): 如果为 True，则频率为对数分布；否则，频率为线性分布。默认为 True。
            input_dim (int): 输入维度，默认为3。
            include_input (bool): 是否包含原始输入张量，默认为 True。
            include_pi (bool): 是否在频率中包含 π，默认为 True。
        """

        super().__init__()

        if logspace:
            # 生成对数空间频率，范围为 [2^0, 2^1, ..., 2^(num_freqs-1)]
            frequencies = 2.0 ** torch.arange(
                num_freqs,
                dtype=torch.float32
            )
        else:
            # 生成线性空间频率，范围为 [1.0, 2^(num_freqs-1)]
            frequencies = torch.linspace(
                1.0,
                2.0 ** (num_freqs - 1),
                num_freqs,
                dtype=torch.float32
            )

        if include_pi:
            # 将频率乘以 π
            frequencies *= torch.pi

        # 将频率张量注册为缓冲区，不作为模型参数保存
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs

        # 计算输出维度
        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim):
        """
        计算输出维度。

        参数:
            input_dim (int): 输入维度。

        返回:
            int: 输出维度。
        """
        # 如果 include_input 为 True 或 num_freqs 为0，则 temp 为1；否则为0
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        # 输出维度为 input_dim * (num_freqs * 2 + temp)
        out_dim = input_dim * (self.num_freqs * 2 + temp)

        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播过程。

        参数:
            x (torch.Tensor): 输入张量，形状为 [..., dim]。

        返回:
            torch.Tensor: 编码后的张量，形状为 [..., dim * (num_freqs * 2 + temp)]，
                          其中 temp 为1如果 include_input 为 True，否则为0。
        """

        if self.num_freqs > 0:
            # 将输入张量扩展一个维度，以便与频率相乘
            embed = (x[..., None].contiguous() * self.frequencies.to(device=x.device, dtype=x.dtype)).view(*x.shape[:-1], -1)
            if self.include_input:
                # 如果包含原始输入，则将原始输入与正弦和余弦嵌入拼接
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                # 否则，仅返回正弦和余弦嵌入
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            # 如果 num_freqs 为0，则返回原始输入
            return x


class CrossAttentionProcessor:
    """
    交叉注意力处理器。该处理器使用缩放点积注意力机制计算查询 (q)、键 (k) 和值 (v) 之间的注意力。
    """
    def __call__(self, attn, q, k, v):
        """
        计算交叉注意力。

        参数:
            attn: 注意力参数（如果有的话）。
            q (torch.Tensor): 查询张量，形状为 (..., seq_len_q, dim).
            k (torch.Tensor): 键张量，形状为 (..., seq_len_k, dim).
            v (torch.Tensor): 值张量，形状为 (..., seq_len_v, dim).

        返回:
            torch.Tensor: 注意力输出，形状为 (..., seq_len_q, dim).
        """
        # 使用 PyTorch 的缩放点积注意力函数计算注意力
        out = F.scaled_dot_product_attention(q, k, v)
        return out


class DropPath(nn.Module):
    """
    DropPath（随机深度）模块，用于在残差块的主路径中按样本随机丢弃路径。
    这种方法通过在训练过程中随机跳过某些层来正则化模型，类似于 Dropout。

    参数:
        drop_prob (float): 丢弃路径的概率，默认为0.0（不丢弃）。
        scale_by_keep (bool): 是否根据保留概率缩放输出，默认为 True。
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        # 丢弃路径的概率
        self.drop_prob = drop_prob
        # 是否根据保留概率缩放输出
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        """
        前向传播过程，按样本随机丢弃路径。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 丢弃路径后的输出张量。
        """
        if self.drop_prob == 0. or not self.training:
            # 如果丢弃概率为0或模型处于评估模式，则直接返回输入
            return x
        # 保留概率
        keep_prob = 1 - self.drop_prob
        # 创建一个与输入张量形状相同的随机张量，除了第一个维度（batch size）外，其余维度均为1
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        # 根据保留概率生成二项分布的随机张量
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            # 如果保留概率大于0且需要缩放，则将随机张量除以保留概率
            random_tensor.div_(keep_prob)
        # 将输入张量与随机张量相乘，实现按样本的随机丢弃
        return x * random_tensor

    def extra_repr(self):
        """
        返回对象的额外表示信息，用于打印模型摘要。

        返回:
            str: 包含丢弃概率的字符串表示。
        """
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class MLP(nn.Module):
    """
    多层感知机（MLP）模块，包含两个线性层和一个GELU激活函数。
    可选的 DropPath 层用于正则化。

    参数:
        width (int): 输入和隐藏层的宽度。
        expand_ratio (int): 隐藏层的扩展比例，默认为4。
        output_width (int, optional): 输出层的宽度。如果为None，则输出宽度与输入宽度相同，默认为None。
        drop_path_rate (float): DropPath 的丢弃概率，默认为0.0（不使用 DropPath）。
    """
    def __init__(
        self, *,
        width: int,
        expand_ratio: int = 4,
        output_width: int = None,
        drop_path_rate: float = 0.0
    ):
        super().__init__()
        # 记录输入宽度
        self.width = width
        # 第一个线性层，扩展比例为4
        self.c_fc = ops.Linear(width, width * expand_ratio)
        # 第二个线性层，输出宽度可选
        self.c_proj = ops.Linear(width * expand_ratio, output_width if output_width is not None else width)
        self.gelu = nn.GELU()
        # 如果 drop_path_rate 大于0，则使用 DropPath，否则使用恒等映射
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        """
        前向传播过程。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: MLP 处理后的输出张量。
        """
        # MLP 的前向传播顺序：线性层 -> GELU -> 线性层 -> DropPath
        return self.drop_path(self.c_proj(self.gelu(self.c_fc(x))))


class QKVMultiheadCrossAttention(nn.Module):
    """
    基于 Q（查询）、K（键）、V（值）的多头交叉注意力模块。
    该模块计算查询和键的点积，并使用 softmax 归一化后加权求和得到输出。

    参数:
        heads (int): 多头注意力的头数。
        width (int, optional): 特征的维度。如果为None，则从其他参数推断，默认为None。
        qk_norm (bool): 是否对查询和键进行层归一化，默认为False。
        norm_layer (nn.Module): 归一化层，默认为 LayerNorm。
    """
    def __init__(
        self,
        *,
        heads: int,
        width=None,
        qk_norm=False,
        norm_layer=ops.LayerNorm
    ):
        super().__init__()
        # 多头注意力的头数
        self.heads = heads
        # 对查询进行归一化
        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        # 对键进行归一化
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        
        # 注意力处理器
        self.attn_processor = CrossAttentionProcessor()

    def forward(self, q, kv):
        """
        前向传播过程，计算多头交叉注意力。

        参数:
            q (torch.Tensor): 查询张量，形状为 (batch_size, n_ctx, width)。
            kv (torch.Tensor): 键和值张量，形状为 (batch_size, n_data, width)。

        返回:
            torch.Tensor: 注意力输出，形状为 (batch_size, n_ctx, width)。
        """
        # 获取查询序列长度
        _, n_ctx, _ = q.shape
        # 获取键值对数据形状
        bs, n_data, width = kv.shape
        # 计算每个头的注意力通道数
        attn_ch = width // self.heads // 2
        # 重塑查询张量为 (batch_size, n_ctx, heads, -1)
        q = q.view(bs, n_ctx, self.heads, -1)
        # 重塑键值对张量为 (batch_size, n_data, heads, -1)
        kv = kv.view(bs, n_data, self.heads, -1)
        # 将键和值分开
        k, v = torch.split(kv, attn_ch, dim=-1)

        # 对查询进行归一化
        q = self.q_norm(q)
        # 对键进行归一化
        k = self.k_norm(k)
        # 重塑张量以适应多头注意力机制
        q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.heads), (q, k, v))
        # 计算注意力输出
        out = self.attn_processor(self, q, k, v)
        # 重塑输出张量
        out = out.transpose(1, 2).reshape(bs, n_ctx, -1)
        return out


class MultiheadCrossAttention(nn.Module):
    """
    多头交叉注意力模块，包含查询、键、值线性变换、注意力计算和输出投影。

    参数:
        width (int): 特征的维度。
        heads (int): 多头注意力的头数。
        qkv_bias (bool): 是否在查询、键、值线性变换中使用偏置，默认为True。
        data_width (int, optional): 数据输入的宽度。如果为None，则与宽度相同，默认为None。
        norm_layer (nn.Module): 归一化层，默认为 LayerNorm。
        qk_norm (bool): 是否对查询和键进行层归一化，默认为False。
        kv_cache (bool): 是否使用键值缓存，默认为False。
    """
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        qkv_bias: bool = True,
        data_width: Optional[int] = None,
        norm_layer=ops.LayerNorm,
        qk_norm: bool = False,
        kv_cache: bool = False,
    ):
        super().__init__()
        # 特征维度
        self.width = width
        # 多头注意力的头数
        self.heads = heads
        # 数据输入的宽度
        self.data_width = width if data_width is None else data_width
        # 查询线性变换
        self.c_q = ops.Linear(width, width, bias=qkv_bias)
        # 键和值线性变换
        self.c_kv = ops.Linear(self.data_width, width * 2, bias=qkv_bias)
        # 输出投影线性变换
        self.c_proj = ops.Linear(width, width)
        # 多头交叉注意力机制
        self.attention = QKVMultiheadCrossAttention(
            heads=heads,
            width=width,
            norm_layer=norm_layer,
            qk_norm=qk_norm
        )
        # 是否使用键值缓存
        self.kv_cache = kv_cache
        # 键值缓存数据
        self.data = None

    def forward(self, x, data):
        """
        前向传播过程，计算多头交叉注意力。

        参数:
            x (torch.Tensor): 查询输入张量，形状为 (batch_size, n_ctx, width)。
            data (torch.Tensor): 数据输入张量，形状为 (batch_size, n_data, data_width)。

        返回:
            torch.Tensor: 注意力输出，形状为 (batch_size, n_ctx, width)。
        """
        # 对查询输入进行线性变换
        x = self.c_q(x)
        if self.kv_cache:
            if self.data is None:
                # 如果使用缓存且缓存为空，则计算键值对并缓存
                self.data = self.c_kv(data)
                logging.info('Save kv cache,this should be called only once for one mesh')
            # 使用缓存的键值对
            data = self.data
        else:
            # 否则，重新计算键值对
            data = self.c_kv(data)
        
        # 计算多头交叉注意力
        x = self.attention(x, data)
        # 对注意力输出进行投影
        x = self.c_proj(x)
        return x


class ResidualCrossAttentionBlock(nn.Module):
    """
    残差交叉注意力块（Residual Cross-Attention Block）。
    该块结合了多头交叉注意力机制和 MLP 层，并通过残差连接进行增强。

    参数:
        width (int): 特征的维度。
        heads (int): 多头注意力的头数。
        mlp_expand_ratio (int): MLP 层的扩展比例，默认为4。
        data_width (int, optional): 数据输入的宽度。如果为 None，则与 `width` 相同，默认为 None。
        qkv_bias (bool): 是否在查询、键、值线性变换中使用偏置，默认为 True。
        norm_layer (nn.Module): 归一化层，默认为 LayerNorm。
        qk_norm (bool): 是否对查询和键进行层归一化，默认为 False。
    """
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        mlp_expand_ratio: int = 4,
        data_width: Optional[int] = None,
        qkv_bias: bool = True,
        norm_layer=ops.LayerNorm,
        qk_norm: bool = False
    ):
        super().__init__()

        if data_width is None:
            # 如果 data_width 未指定，则默认为 width
            data_width = width

        # 多头交叉注意力层
        self.attn = MultiheadCrossAttention(
            width=width,
            heads=heads,
            data_width=data_width,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            qk_norm=qk_norm
        )
        # 第一个归一化层，用于输入 `x`
        self.ln_1 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        # 第二个归一化层，用于输入 `data`
        self.ln_2 = norm_layer(data_width, elementwise_affine=True, eps=1e-6)
        # 第三个归一化层，用于 MLP 的输入
        self.ln_3 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        # MLP 层，用于进一步处理 `x`
        self.mlp = MLP(width=width, expand_ratio=mlp_expand_ratio)

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        """
        前向传播过程。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, n_ctx, width)。
            data (torch.Tensor): 数据输入张量，形状为 (batch_size, n_data, data_width)。

        返回:
            torch.Tensor: 处理后的输出张量，形状为 (batch_size, n_ctx, width)。
        """
        # 残差连接1: 注意力输出 + 原始输入 `x`
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        # 残差连接2: MLP 输出 + 注意力处理后的 `x`
        x = x + self.mlp(self.ln_3(x))
        return x


class QKVMultiheadAttention(nn.Module):
    """
    基于 Q（查询）、K（键）、V（值）的多头自注意力模块。
    该模块将输入张量 `qkv` 分割为查询、键和值，并计算缩放点积注意力。

    参数:
        heads (int): 多头注意力的头数。
        width (int, optional): 特征的维度。如果为 None，则从其他参数推断，默认为 None。
        qk_norm (bool): 是否对查询和键进行层归一化，默认为 False。
        norm_layer (nn.Module): 归一化层，默认为 LayerNorm。
    """
    def __init__(
        self,
        *,
        heads: int,
        width=None,
        qk_norm=False,
        norm_layer=ops.LayerNorm
    ):
        super().__init__()
        # 多头注意力的头数
        self.heads = heads
        # 对查询进行归一化
        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        # 对键进行归一化
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()

    def forward(self, qkv):
        """
        前向传播过程，计算多头自注意力。

        参数:
            qkv (torch.Tensor): 输入张量，形状为 (batch_size, n_ctx, width)。

        返回:
            torch.Tensor: 注意力输出，形状为 (batch_size, n_ctx, width)。
        """
        # 获取输入张量的形状
        bs, n_ctx, width = qkv.shape
        # 计算每个头的注意力通道数
        attn_ch = width // self.heads // 3
        # 重塑张量为 (batch_size, n_ctx, heads, -1)
        qkv = qkv.view(bs, n_ctx, self.heads, -1)

        # 将输入分割为查询、键和值
        q, k, v = torch.split(qkv, attn_ch, dim=-1)

        # 对查询进行归一化
        q = self.q_norm(q)
        # 对键进行归一化
        k = self.k_norm(k)

        # 重塑张量以适应多头注意力机制
        q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.heads), (q, k, v))
        # 计算缩放点积注意力
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bs, n_ctx, -1)
        return out


class MultiheadAttention(nn.Module):
    """
    多头自注意力模块，包含查询、键、值线性变换、注意力计算和输出投影。

    参数:
        width (int): 特征的维度。
        heads (int): 多头注意力的头数。
        qkv_bias (bool): 是否在查询、键、值线性变换中使用偏置，默认为 True。
        norm_layer (nn.Module): 归一化层，默认为 LayerNorm。
        qk_norm (bool): 是否对查询和键进行层归一化，默认为 False。
        drop_path_rate (float): DropPath 的丢弃概率，默认为 0.0（不使用 DropPath）。
    """
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        qkv_bias: bool,
        norm_layer=ops.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0
    ):
        super().__init__()
        # 特征维度
        self.width = width
        # 多头注意力的头数
        self.heads = heads
        # 查询、键、值线性变换
        self.c_qkv = ops.Linear(width, width * 3, bias=qkv_bias)
        # 输出投影线性变换
        self.c_proj = ops.Linear(width, width)
        # 多头自注意力机制
        self.attention = QKVMultiheadAttention(
            heads=heads,
            width=width,
            norm_layer=norm_layer,
            qk_norm=qk_norm
        )
        # DropPath 层
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        """
        前向传播过程，计算多头自注意力。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, n_ctx, width)。

        返回:
            torch.Tensor: 注意力输出，形状为 (batch_size, n_ctx, width)。
        """
        # 对输入进行线性变换，得到查询、键、值
        x = self.c_qkv(x)
        # 计算多头自注意力
        x = self.attention(x)
        # 应用 DropPath 和输出投影
        x = self.drop_path(self.c_proj(x))
        return x


class ResidualAttentionBlock(nn.Module):
    """
    残差自注意力块（Residual Attention Block）。
    该块结合了多头自注意力机制和 MLP 层，并通过残差连接进行增强。

    参数:
        width (int): 特征的维度。
        heads (int): 多头注意力的头数。
        qkv_bias (bool): 是否在查询、键、值线性变换中使用偏置，默认为 True。
        norm_layer (nn.Module): 归一化层，默认为 LayerNorm。
        qk_norm (bool): 是否对查询和键进行层归一化，默认为 False。
        drop_path_rate (float): DropPath 的丢弃概率，默认为 0.0（不使用 DropPath）。
    """
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        qkv_bias: bool = True,
        norm_layer=ops.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        # 多头自注意力层
        self.attn = MultiheadAttention(
            width=width,
            heads=heads,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate
        )
        # 第一个归一化层，用于输入 `x`
        self.ln_1 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        # MLP 层，用于进一步处理 `x`
        self.mlp = MLP(width=width, drop_path_rate=drop_path_rate)
        # 第二个归一化层，用于 MLP 的输入
        self.ln_2 = norm_layer(width, elementwise_affine=True, eps=1e-6)

    def forward(self, x: torch.Tensor):
        """
        前向传播过程。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, n_ctx, width)。

        返回:
            torch.Tensor: 处理后的输出张量，形状为 (batch_size, n_ctx, width)。
        """
        # 残差连接1: 注意力输出 + 原始输入 `x`
        x = x + self.attn(self.ln_1(x))
        # 残差连接2: MLP 输出 + 注意力处理后的 `x`
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """
    Transformer 模型，包含多个残差注意力块（ResidualAttentionBlock）。
    该模型通过堆叠多个自注意力层和 MLP 层来捕捉输入数据的复杂依赖关系。

    参数:
        width (int): 特征的维度。
        layers (int): Transformer 层的数量。
        heads (int): 多头注意力的头数。
        qkv_bias (bool): 是否在查询、键、值线性变换中使用偏置，默认为 True。
        norm_layer (nn.Module): 归一化层，默认为 LayerNorm。
        qk_norm (bool): 是否对查询和键进行层归一化，默认为 False。
        drop_path_rate (float): DropPath 的丢弃概率，默认为 0.0（不使用 DropPath）。
    """
    def __init__(
        self,
        *,
        width: int,
        layers: int,
        heads: int,
        qkv_bias: bool = True,
        norm_layer=ops.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0
    ):
        super().__init__()
        # Transformer 的特征维度
        self.width = width
        # Transformer 层的数量
        self.layers = layers

        # 使用 ModuleList 存储多个残差注意力块
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    width=width,
                    heads=heads,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    qk_norm=qk_norm,
                    drop_path_rate=drop_path_rate
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        """
        前向传播过程，将输入张量通过所有残差注意力块。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, n_ctx, width)。

        返回:
            torch.Tensor: Transformer 处理后的输出张量，形状为 (batch_size, n_ctx, width)。
        """
        for block in self.resblocks:
            # 将输入传递给每个残差注意力块
            x = block(x)
        return x


class CrossAttentionDecoder(nn.Module):
    """
    交叉注意力解码器（Cross-Attention Decoder），用于将潜在向量解码为3D体积数据。
    该解码器结合了傅里叶嵌入、交叉注意力机制和 MLP 层。

    参数:
        out_channels (int): 输出通道数。
        fourier_embedder (FourierEmbedder): 傅里叶嵌入器，用于对查询点进行嵌入。
        width (int): 特征的维度。
        heads (int): 多头注意力的头数。
        mlp_expand_ratio (int): MLP 层的扩展比例，默认为4。
        downsample_ratio (int): 下采样比例，默认为1（不进行下采样）。
        enable_ln_post (bool): 是否在输出投影后应用层归一化，默认为 True。
        qkv_bias (bool): 是否在查询、键、值线性变换中使用偏置，默认为 True。
        qk_norm (bool): 是否对查询和键进行层归一化，默认为 False。
        label_type (str): 标签类型，默认为 "binary"。
    """
    def __init__(
        self,
        *,
        out_channels: int,
        fourier_embedder: FourierEmbedder,
        width: int,
        heads: int,
        mlp_expand_ratio: int = 4,
        downsample_ratio: int = 1,
        enable_ln_post: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        label_type: str = "binary"
    ):
        super().__init__()

        # 是否启用后置层归一化
        self.enable_ln_post = enable_ln_post
        # 傅里叶嵌入器
        self.fourier_embedder = fourier_embedder
        # 下采样比例
        self.downsample_ratio = downsample_ratio
        # 线性层，用于将嵌入后的查询点投影到指定宽度
        self.query_proj = ops.Linear(self.fourier_embedder.out_dim, width)

        if self.downsample_ratio != 1:
            # 如果进行下采样，则应用线性层进行投影
            self.latents_proj = ops.Linear(width * downsample_ratio, width)
        if self.enable_ln_post == False:
            # 如果不启用后置层归一化，则不进行 QK 归一化
            qk_norm = False

        # 交叉注意力解码器块
        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            width=width,
            mlp_expand_ratio=mlp_expand_ratio,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm
        )

        if self.enable_ln_post:
            # 后置层归一化层
            self.ln_post = ops.LayerNorm(width)
        # 输出投影层，将宽度映射到输出通道数
        self.output_proj = ops.Linear(width, out_channels)
        # 标签类型
        self.label_type = label_type
        # 计数器，用于记录处理过的查询点数量
        self.count = 0

    def forward(self, queries=None, query_embeddings=None, latents=None):
        """
        前向传播过程，将查询点解码为3D体积数据。

        参数:
            queries (torch.Tensor, optional): 查询点张量，形状为 (batch_size, n_points, 3)，默认为 None。
            query_embeddings (torch.Tensor, optional): 预先嵌入的查询点张量，形状为 (batch_size, n_points, width)，默认为 None。
            latents (torch.Tensor, optional): 潜在向量张量，形状为 (batch_size, latent_dim)，默认为 None。

        返回:
            torch.Tensor: 解码后的3D体积数据，形状为 (batch_size, n_points, out_channels)。
        """
        if query_embeddings is None:
            # 如果未提供预先嵌入的查询点，则使用傅里叶嵌入器进行嵌入，并投影到指定宽度
            query_embeddings = self.query_proj(self.fourier_embedder(queries).to(latents.dtype))
        # 增加已处理的查询点数量
        self.count += query_embeddings.shape[1]

        if self.downsample_ratio != 1:
            # 如果进行下采样，则应用线性层进行投影
            latents = self.latents_proj(latents)

        # 通过交叉注意力解码器块处理查询嵌入和潜在向量
        x = self.cross_attn_decoder(query_embeddings, latents)
        if self.enable_ln_post:
            # 如果启用后置层归一化，则应用层归一化
            x = self.ln_post(x)
        # 通过输出投影层得到最终的输出
        occ = self.output_proj(x)

        # 返回解码后的3D体积数据
        return occ


class ShapeVAE(nn.Module):
    """
    ShapeVAE 模型，结合了 Transformer 和交叉注意力解码器，用于3D形状生成或重建。

    ShapeVAE 模型通过潜在向量（latents）来表示3D形状，并通过解码器将这些潜在向量转换为3D体积数据。
    该模型使用傅里叶嵌入来捕捉多频域信息，并结合 Transformer 和交叉注意力机制来捕捉复杂的空间依赖关系。

    参数:
        embed_dim (int): 潜在向量的嵌入维度。
        width (int): 特征的维度，用于 Transformer 和解码器。
        heads (int): 多头注意力的头数，用于 Transformer 和解码器。
        num_decoder_layers (int): Transformer 解码器的层数。
        geo_decoder_downsample_ratio (int, optional): 解码器中下采样的比例，默认为1（不进行下采样）。
        geo_decoder_mlp_expand_ratio (int, optional): 解码器中 MLP 层的扩展比例，默认为4。
        geo_decoder_ln_post (bool, optional): 是否在解码器输出后应用层归一化，默认为 True。
        num_freqs (int, optional): 傅里叶嵌入器的频率数量，默认为8。
        include_pi (bool, optional): 是否在傅里叶嵌入中包含 π，默认为 True。
        qkv_bias (bool, optional): 是否在查询、键、值线性变换中使用偏置，默认为 True。
        qk_norm (bool, optional): 是否对查询和键进行层归一化，默认为 False。
        label_type (str, optional): 标签类型，默认为 "binary"。
        drop_path_rate (float, optional): DropPath 的丢弃概率，默认为0.0。
        scale_factor (float, optional): 缩放因子，默认为1.0，用于调整输出尺度。
    """
    def __init__(
        self,
        *,
        embed_dim: int,
        width: int,
        heads: int,
        num_decoder_layers: int,
        geo_decoder_downsample_ratio: int = 1,
        geo_decoder_mlp_expand_ratio: int = 4,
        geo_decoder_ln_post: bool = True,
        num_freqs: int = 8,
        include_pi: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        label_type: str = "binary",
        drop_path_rate: float = 0.0,
        scale_factor: float = 1.0,
    ):
        super().__init__()
        # 是否在解码器输出后应用层归一化
        self.geo_decoder_ln_post = geo_decoder_ln_post

        # 初始化傅里叶嵌入器，用于对查询点进行多频域嵌入
        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)

        # 初始化一个线性层，用于将潜在向量从 embed_dim 映射到 width
        self.post_kl = ops.Linear(embed_dim, width)

        # 初始化 Transformer 模型，用于处理潜在向量
        self.transformer = Transformer(
            width=width,
            layers=num_decoder_layers,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate
        )

        # 初始化交叉注意力解码器，用于将 Transformer 输出的潜在向量解码为3D体积数据
        self.geo_decoder = CrossAttentionDecoder(
            fourier_embedder=self.fourier_embedder,
            out_channels=1,  # 输出通道数为1，表示二值化的3D体积数据
            mlp_expand_ratio=geo_decoder_mlp_expand_ratio,  # MLP 层的扩展比例
            downsample_ratio=geo_decoder_downsample_ratio,  # 下采样比例
            enable_ln_post=self.geo_decoder_ln_post,  # 是否启用后置层归一化
            width=width // geo_decoder_downsample_ratio,  # 根据下采样比例调整特征维度
            heads=heads // geo_decoder_downsample_ratio,  # 根据下采样比例调整多头注意力的头数
            qkv_bias=qkv_bias,  # 是否使用偏置
            qk_norm=qk_norm,  # 是否进行 QK 归一化
            label_type=label_type,  # 标签类型
        )

        # 初始化体积解码器，用于将交叉注意力解码器的输出转换为3D体积数据
        self.volume_decoder = VanillaVolumeDecoder()
        # 初始化缩放因子，用于调整输出尺度
        self.scale_factor = scale_factor

    def decode(self, latents, **kwargs):
        """
        解码过程，将潜在向量解码为3D体积数据。

        参数:
            latents (torch.Tensor): 输入的潜在向量，形状为 (batch_size, latent_dim)。
            **kwargs: 其他可选的关键字参数，包括:
                - bounds (float or tuple): 边界框范围，默认为1.01。
                - num_chunks (int): 每个批次的查询点数量，默认为8000。
                - octree_resolution (int): 八叉树分辨率，默认为256。
                - enable_pbar (bool): 是否启用进度条，默认为True。

        返回:
            torch.Tensor: 解码后的3D体积数据，形状为 (batch_size, N, N, N)。
        """
        # 对潜在向量进行后处理，通过线性层将维度从 latent_dim 映射到 width
        latents = self.post_kl(latents.movedim(-2, -1))  # (batch_size, width, ...)

        # 通过 Transformer 模型处理后处理的潜在向量
        latents = self.transformer(latents)  # (batch_size, width, ...)

        # 获取解码参数，包括边界框范围、每个批次的查询点数量、八叉树分辨率和是否启用进度条
        bounds = kwargs.get("bounds", 1.01)  # 获取边界框范围，默认为1.01
        num_chunks = kwargs.get("num_chunks", 8000)  # 获取每个批次的查询点数量，默认为8000
        octree_resolution = kwargs.get("octree_resolution", 256)  # 获取八叉树分辨率，默认为256
        enable_pbar = kwargs.get("enable_pbar", True)  # 是否启用进度条，默认为True

        # 使用体积解码器将潜在向量解码为3D体积数据
        grid_logits = self.volume_decoder(latents, self.geo_decoder, bounds=bounds, num_chunks=num_chunks, octree_resolution=octree_resolution, enable_pbar=enable_pbar)
        
        # 返回解码后的3D体积数据，并调整维度顺序
        return grid_logits.movedim(-2, -1)

    def encode(self, x):
        """
        编码过程，将输入3D数据编码为潜在向量。

        参数:
            x (torch.Tensor): 输入的3D数据。

        返回:
            torch.Tensor: 编码后的潜在向量。如果未实现，则返回 None。
        """
        return None
