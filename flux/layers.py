import math
from dataclasses import dataclass
import torch
from torch import Tensor, nn

from math import attention, rope
import ops
import common_dit


class EmbedND(nn.Module):
    """
    EmbedND 类，用于生成多维嵌入。
    该类对输入的多维索引应用旋转位置编码（RoPE），并生成最终的嵌入张量。
    """
    def __init__(self, dim: int, theta: int, axes_dim: list):
        """
        初始化 EmbedND 模块。

        参数:
            dim (int): 嵌入的维度。
            theta (int): RoPE 的旋转角度参数。
            axes_dim (list): 每个轴的维度列表，用于确定每个轴的嵌入大小。
        """
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        """
        前向传播函数，生成嵌入张量。

        参数:
            ids (torch.Tensor): 输入的多维索引张量，形状为 (..., n_axes)。

        返回:
            torch.Tensor: 应用了 RoPE 后的嵌入张量，形状为 (..., dim)。
        """
        # 获取轴的数量
        n_axes = ids.shape[-1]
        # 对每个轴应用 RoPE，并沿最后一个维度拼接结果
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        # 在倒数第二个维度上添加一个维度
        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    生成正弦时间步嵌入。

    参数:
        t (torch.Tensor): 一个 1-D 张量，包含 N 个索引，每个批次元素一个。这些可能是分数。
        dim (int): 输出的维度。
        max_period (int, 可选): 控制嵌入的最小频率，默认为 10000。
        time_factor (float, 可选): 时间因子，用于缩放时间步，默认为 1000.0。

    返回:
        torch.Tensor: 一个 (N, D) 的位置嵌入张量。
    """
    # 缩放时间步
    t = time_factor * t
    # 计算一半的维度
    half = dim // 2
    # 计算频率
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
    
    # 计算角度
    args = t[:, None].float() * freqs[None]
    # 计算余弦和正弦，并拼接
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        # 如果维度为奇数，添加零填充
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        # 确保嵌入与输入张量具有相同的数据类型
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    """
    MLPEmbedder 类，使用多层感知机（MLP）生成嵌入。
    该类通过线性层和 SiLU 激活函数处理输入向量，生成嵌入向量。
    """
    def __init__(self, in_dim: int, hidden_dim: int, dtype=None, device=None, operations=None):
        """
        初始化 MLPEmbedder 模块。

        参数:
            in_dim (int): 输入向量的维度。
            hidden_dim (int): 隐藏层的维度。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
            operations (module, 可选): 包含自定义操作的模块，如 Linear，默认为 None。
        """
        super().__init__()
        # 定义输入线性层
        self.in_layer = operations.Linear(in_dim, hidden_dim, bias=True, dtype=dtype, device=device)
        # 定义 SiLU 激活函数
        self.silu = nn.SiLU()
        # 定义输出线性层
        self.out_layer = operations.Linear(hidden_dim, hidden_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播函数，生成嵌入向量。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 生成的嵌入向量。
        """
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    """
    RMSNorm 类，实现均方根归一化（Root Mean Square Normalization）。
    该类对输入张量进行归一化处理。
    """
    def __init__(self, dim: int, dtype=None, device=None, operations=None):
        """
        初始化 RMSNorm 模块。

        参数:
            dim (int): 输入张量的维度。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
            operations (module, 可选): 包含自定义操作的模块，默认为 None。
        """
        super().__init__()
        # 定义缩放因子参数
        self.scale = nn.Parameter(torch.empty((dim), dtype=dtype, device=device))

    def forward(self, x: Tensor):
        """
        前向传播函数，应用 RMS 归一化。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 归一化后的张量。
        """
        return common_dit.rms_norm(x, self.scale, 1e-6)


class QKNorm(torch.nn.Module):
    """
    QKNorm 类，实现查询和键的归一化。
    该类对输入的查询和键张量应用 RMS 归一化。
    """
    def __init__(self, dim: int, dtype=None, device=None, operations=None):
        """
        初始化 QKNorm 模块。

        参数:
            dim (int): 输入张量的维度。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
            operations (module, 可选): 包含自定义操作的模块，默认为 None。
        """
        super().__init__()
        # 定义查询和键的 RMS 归一化层
        self.query_norm = RMSNorm(dim, dtype=dtype, device=device, operations=operations)
        self.key_norm = RMSNorm(dim, dtype=dtype, device=device, operations=operations)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple:
        """
        前向传播函数，应用查询和键的归一化。

        参数:
            q (torch.Tensor): 查询张量。
            k (torch.Tensor): 键张量。
            v (torch.Tensor): 值张量。

        返回:
            tuple: 归一化后的查询和键张量。
        """
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    """
    SelfAttention 类，实现自注意力机制。
    该类通过线性层、归一化层和投影层实现自注意力机制。
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, dtype=None, device=None, operations=None):
        """
        初始化 SelfAttention 模块。

        参数:
            dim (int): 输入和输出的维度。
            num_heads (int, 可选): 注意力头的数量，默认为 8。
            qkv_bias (bool, 可选): 查询、键和值线性层是否使用偏置，默认为 False。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
            operations (module, 可选): 包含自定义操作的模块，默认为 None。
        """
        super().__init__()
        self.num_heads = num_heads
        # 计算每个头的维度
        head_dim = dim // num_heads

        # 定义查询、键和值线性层
        self.qkv = operations.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        # 定义查询和键的归一化层
        self.norm = QKNorm(head_dim, dtype=dtype, device=device, operations=operations)
        # 定义投影线性层
        self.proj = operations.Linear(dim, dim, dtype=dtype, device=device)


@dataclass
class ModulationOut:
    """
    ModulationOut 数据类，用于存储调制输出。
    该类包含调制后的移位、缩放和门控张量。
    """
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    """
    Modulation 类，实现调制机制。
    该类通过线性层和 SiLU 激活函数对输入向量进行调制，生成移位、缩放和门控张量。
    """
    def __init__(self, dim: int, double: bool, dtype=None, device=None, operations=None):
        """
        初始化 Modulation 模块。

        参数:
            dim (int): 输入向量的维度。
            double (bool): 是否进行双重调制。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
            operations (module, 可选): 包含自定义操作的模块，默认为 None。
        """
        super().__init__()
        self.is_double = double
        # 根据是否双重调制设置乘数
        self.multiplier = 6 if double else 3
        # 定义线性层
        self.lin = operations.Linear(dim, self.multiplier * dim, bias=True, dtype=dtype, device=device)

    def forward(self, vec: Tensor) -> tuple:
        """
        前向传播函数，应用调制。

        参数:
            vec (torch.Tensor): 输入张量。

        返回:
            tuple: 包含调制后的 ModulationOut 对象。
        """
        if vec.ndim == 2:
            # 如果输入是二维张量，则添加一个维度
            vec = vec[:, None, :]
        # 应用线性层和 SiLU 激活函数
        out = self.lin(nn.functional.silu(vec)).chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


def apply_mod(tensor, m_mult, m_add=None, modulation_dims=None):
    """
    对输入张量应用调制（缩放和平移）。

    参数:
        tensor (torch.Tensor): 输入张量。
        m_mult (torch.Tensor): 缩放因子张量。
        m_add (torch.Tensor, 可选): 平移因子张量，默认为 None。
        modulation_dims (list of tuples, 可选): 要应用调制的维度范围，格式为 [(start_dim, end_dim, index), ...]，默认为 None。

    返回:
        torch.Tensor: 调制后的张量。
    """
    if modulation_dims is None:
        # 如果没有指定调制维度，则对整个张量应用缩放和平移
        if m_add is not None:
            return tensor * m_mult + m_add
        else:
            return tensor * m_mult
    else:
        # 如果指定了调制维度，则对指定范围内的张量应用缩放和平移
        for d in modulation_dims:
            tensor[:, d[0]:d[1]] *= m_mult[:, d[2]]
            if m_add is not None:
                tensor[:, d[0]:d[1]] += m_add[:, d[2]]
        return tensor


class DoubleStreamBlock(nn.Module):
    """
    双流块（DoubleStreamBlock）类。
    该类实现了图像和文本的双流处理，每个流包含调制、归一化、自注意力机制和 MLP。
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False, flipped_img_txt=False, dtype=None, device=None, operations=None):
        """
        初始化双流块。

        参数:
            hidden_size (int): 隐藏层的大小。
            num_heads (int): 自注意力机制中头的数量。
            mlp_ratio (float): MLP 中隐藏层大小的比例。
            qkv_bias (bool, 可选): 查询、键和值线性层是否使用偏置，默认为 False。
            flipped_img_txt (bool, 可选): 是否翻转图像和文本流的顺序，默认为 False。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
            operations (module, 可选): 包含自定义操作的模块，默认为 None。
        """
        super().__init__()

        # 计算 MLP 隐藏层大小
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        # 定义图像流的调制模块
        self.img_mod = Modulation(hidden_size, double=True, dtype=dtype, device=device, operations=operations)
        # 定义图像流的归一化层
        self.img_norm1 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        # 定义图像流的自注意力机制
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, dtype=dtype, device=device, operations=operations)
        
        # 定义图像流的第二个归一化层
        self.img_norm2 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        # 定义图像流的 MLP
        self.img_mlp = nn.Sequential(
            operations.Linear(hidden_size, mlp_hidden_dim, bias=True, dtype=dtype, device=device),
            nn.GELU(approximate="tanh"),
            operations.Linear(mlp_hidden_dim, hidden_size, bias=True, dtype=dtype, device=device),
        )

        # 定义文本流的调制模块
        self.txt_mod = Modulation(hidden_size, double=True, dtype=dtype, device=device, operations=operations)
        # 定义文本流的归一化层
        self.txt_norm1 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        # 定义文本流的自注意力机制
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, dtype=dtype, device=device, operations=operations)

        # 定义文本流的第二个归一化层
        self.txt_norm2 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        # 定义文本流的 MLP
        self.txt_mlp = nn.Sequential(
            operations.Linear(hidden_size, mlp_hidden_dim, bias=True, dtype=dtype, device=device),
            nn.GELU(approximate="tanh"),
            operations.Linear(mlp_hidden_dim, hidden_size, bias=True, dtype=dtype, device=device),
        )

        # 是否翻转图像和文本流的顺序
        self.flipped_img_txt = flipped_img_txt

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, attn_mask=None, modulation_dims_img=None, modulation_dims_txt=None):
        """
        前向传播函数，处理图像和文本输入。

        参数:
            img (torch.Tensor): 输入图像张量。
            txt (torch.Tensor): 输入文本张量。
            vec (torch.Tensor): 条件向量张量，用于调制。
            pe (torch.Tensor): 位置嵌入张量。
            attn_mask (torch.Tensor, 可选): 注意力掩码，默认为 None。
            modulation_dims_img (list of tuples, 可选): 图像流调制维度，默认为 None。
            modulation_dims_txt (list of tuples, 可选): 文本流调制维度，默认为 None。

        返回:
            tuple: 包含处理后的图像和文本张量。
        """
        # 应用调制到图像和文本流
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # 准备图像流进行注意力机制处理
        img_modulated = self.img_norm1(img)
        img_modulated = apply_mod(img_modulated, (1 + img_mod1.scale), img_mod1.shift, modulation_dims_img)
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # 准备文本流进行注意力机制处理
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = apply_mod(txt_modulated, (1 + txt_mod1.scale), txt_mod1.shift, modulation_dims_txt)
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # 根据是否翻转图像和文本流的顺序执行注意力机制
        if self.flipped_img_txt:
            # run actual attention
            attn = attention(torch.cat((img_q, txt_q), dim=2),
                             torch.cat((img_k, txt_k), dim=2),
                             torch.cat((img_v, txt_v), dim=2),
                             pe=pe, mask=attn_mask)

            img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1]:]
        else:
            # run actual attention
            attn = attention(torch.cat((txt_q, img_q), dim=2),
                             torch.cat((txt_k, img_k), dim=2),
                             torch.cat((txt_v, img_v), dim=2),
                             pe=pe, mask=attn_mask)

            txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1]:]

        # 计算图像流块
        img = img + apply_mod(self.img_attn.proj(img_attn), img_mod1.gate, None, modulation_dims_img)
        img = img + apply_mod(self.img_mlp(apply_mod(self.img_norm2(img), (1 + img_mod2.scale), img_mod2.shift, modulation_dims_img)), img_mod2.gate, None, modulation_dims_img)

        # 计算文本流块
        txt += apply_mod(self.txt_attn.proj(txt_attn), txt_mod1.gate, None, modulation_dims_txt)
        txt += apply_mod(self.txt_mlp(apply_mod(self.txt_norm2(txt), (1 + txt_mod2.scale), txt_mod2.shift, modulation_dims_txt)), txt_mod2.gate, None, modulation_dims_txt)

        # 如果文本张量的数据类型为 float16，则处理 NaN 和无穷大值
        if txt.dtype == torch.float16:
            txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

        return img, txt


class SingleStreamBlock(nn.Module):
    """
    单流块（SingleStreamBlock）类。
    该类实现了 DiT 块，包含了并行线性层和调制接口。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float = None,
        dtype=None,
        device=None,
        operations=None
    ):
        """
        初始化单流块。

        参数:
            hidden_size (int): 隐藏层的大小。
            num_heads (int): 自注意力机制中头的数量。
            mlp_ratio (float, 可选): MLP 中隐藏层大小的比例，默认为 4.0。
            qk_scale (float, 可选): QK 缩放因子，默认为 None。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
            operations (module, 可选): 包含自定义操作的模块，默认为 None。
        """
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # 定义 qkv 和 mlp_in 的线性层
        self.linear1 = operations.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim, dtype=dtype, device=device)
        # 定义 proj 和 mlp_out 的线性层
        self.linear2 = operations.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, dtype=dtype, device=device)

        self.norm = QKNorm(head_dim, dtype=dtype, device=device, operations=operations)

        self.hidden_size = hidden_size
        # 定义预归一化层
        self.pre_norm = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)

        # 定义 MLP 激活函数
        self.mlp_act = nn.GELU(approximate="tanh")
        # 定义调制模块
        self.modulation = Modulation(hidden_size, double=False, dtype=dtype, device=device, operations=operations)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor, attn_mask=None, modulation_dims=None) -> Tensor:
        """
        前向传播函数，处理输入张量。

        参数:
            x (torch.Tensor): 输入张量。
            vec (torch.Tensor): 条件向量张量，用于调制。
            pe (torch.Tensor): 位置嵌入张量。
            attn_mask (torch.Tensor, 可选): 注意力掩码，默认为 None。
            modulation_dims (list of tuples, 可选): 调制维度，默认为 None。

        返回:
            torch.Tensor: 处理后的输出张量。
        """
        mod, _ = self.modulation(vec)
        # 应用调制到预归一化后的输入张量，并分割为 qkv 和 mlp 部分
        qkv, mlp = torch.split(self.linear1(apply_mod(self.pre_norm(x), (1 + mod.scale), mod.shift, modulation_dims)), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        # 重塑 qkv 张量以分离查询、键和值
        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # 应用归一化到查询和键
        q, k = self.norm(q, k, v)

        # 计算注意力机制
        attn = attention(q, k, v, pe=pe, mask=attn_mask)
        # 计算 MLP 激活函数，并连接注意力机制输出和 MLP 输出，然后通过第二个线性层
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        # 应用调制到输出张量，并将其加到输入张量上
        x += apply_mod(output, mod.gate, None, modulation_dims)
        # 如果张量的数据类型为 float16，则处理 NaN 和无穷大值
        if x.dtype == torch.float16:
            x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
        return x


class LastLayer(nn.Module):
    """
    最后一层（LastLayer）类。
    该类实现了模型的最后一层，包含归一化、调制和线性变换。
    """
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, dtype=None, device=None, operations=None):
        """
        初始化最后一层。

        参数:
            hidden_size (int): 隐藏层的大小。
            patch_size (int): 补丁的大小。
            out_channels (int): 输出通道的数量。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
            operations (module, 可选): 包含自定义操作的模块，默认为 None。
        """
        super().__init__()
        # 定义最终的归一化层
        self.norm_final = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        # 定义线性变换层
        self.linear = operations.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True, dtype=dtype, device=device)
        # 定义自适应归一化调制模块
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), operations.Linear(hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device))

    def forward(self, x: Tensor, vec: Tensor, modulation_dims=None) -> Tensor:
        """
        前向传播函数，处理输入张量。

        参数:
            x (torch.Tensor): 输入张量。
            vec (torch.Tensor): 条件向量张量，用于调制。
            modulation_dims (list of tuples, 可选): 调制维度，默认为 None。

        返回:
            torch.Tensor: 处理后的输出张量。
        """
        if vec.ndim == 2:
            # 如果 vec 是二维张量，则添加一个维度
            vec = vec[:, None, :]

        # 应用自适应归一化调制
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        x = apply_mod(self.norm_final(x), (1 + scale), shift, modulation_dims)
        # 应用线性变换
        x = self.linear(x)
        return x
