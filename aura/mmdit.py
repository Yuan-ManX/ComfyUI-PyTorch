#AuraFlow MMDiT
#Originally written by the AuraFlow Authors

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.attention import optimized_attention
import ops
import common_dit


def modulate(x, shift, scale):
    """
    应用调制（缩放和平移）到输入张量。

    参数:
        x (torch.Tensor): 输入张量。
        shift (torch.Tensor): 平移因子张量。
        scale (torch.Tensor): 缩放因子张量。

    返回:
        torch.Tensor: 应用调制后的张量。
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def find_multiple(n: int, k: int) -> int:
    """
    找到大于或等于 n 的最小 k 的倍数。

    参数:
        n (int): 目标数。
        k (int): 倍数。

    返回:
        int: 大于或等于 n 的最小 k 的倍数。
    """
    if n % k == 0:
        # 如果 n 已经是 k 的倍数，则返回 n
        return n
    return n + k - (n % k)


class MLP(nn.Module):
    """
    MLP 类，实现了一个多层感知机（MLP）。
    该 MLP 包含两个线性层和一个投影层，使用 SiLU 激活函数。
    """
    def __init__(self, dim, hidden_dim=None, dtype=None, device=None, operations=None) -> None:
        """
        初始化 MLP 模块。

        参数:
            dim (int): 输入和输出的维度。
            hidden_dim (int, 可选): 隐藏层的维度，默认为 4 * dim。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
            operations (module, 可选): 包含自定义操作的模块，默认为 None。
        """
        super().__init__()
        if hidden_dim is None:
            # 如果未指定隐藏层维度，则设置为 4 * dim
            hidden_dim = 4 * dim

        # 计算隐藏层的维度
        n_hidden = int(2 * hidden_dim / 3)
        # 将隐藏层维度调整为 256 的倍数
        n_hidden = find_multiple(n_hidden, 256)

        # 第一个线性层
        self.c_fc1 = operations.Linear(dim, n_hidden, bias=False, dtype=dtype, device=device)
        # 第二个线性层
        self.c_fc2 = operations.Linear(dim, n_hidden, bias=False, dtype=dtype, device=device)
        # 投影层
        self.c_proj = operations.Linear(n_hidden, dim, bias=False, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，应用 MLP。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 输出张量。
        """
        # 应用第一个线性层和 SiLU 激活函数，然后与第二个线性层的输出相乘
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        # 应用投影层
        x = self.c_proj(x)
        # 返回输出张量
        return x


class MultiHeadLayerNorm(nn.Module):
    """
    MultiHeadLayerNorm 类，实现多头层归一化。
    该类对多头注意力机制的输入进行归一化。
    """
    def __init__(self, hidden_size=None, eps=1e-5, dtype=None, device=None):
        """
        初始化 MultiHeadLayerNorm 模块。

        参数:
            hidden_size (int, 可选): 隐藏层的大小，默认为 None。
            eps (float, 可选): 用于数值稳定的 epsilon，默认为 1e-5。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
        """
        
        super().__init__()
        # 初始化缩放参数
        self.weight = nn.Parameter(torch.empty(hidden_size, dtype=dtype, device=device))
        # 设置 epsilon
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """
        前向传播函数，应用多头层归一化。

        参数:
            hidden_states (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 归一化后的张量。
        """
        # 保存输入张量的数据类型
        input_dtype = hidden_states.dtype
        # 将输入张量转换为 float32 以提高数值稳定性
        hidden_states = hidden_states.to(torch.float32)
        # 计算均值
        mean = hidden_states.mean(-1, keepdim=True)
        # 计算方差
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        # 应用归一化
        hidden_states = (hidden_states - mean) * torch.rsqrt(
            variance + self.variance_epsilon
        )
        # 应用缩放
        hidden_states = self.weight.to(torch.float32) * hidden_states
        # 返回归一化后的张量，并转换回原始数据类型
        return hidden_states.to(input_dtype)


class SingleAttention(nn.Module):
    """
    SingleAttention 类，实现单头自注意力机制。
    该类实现了单头自注意力机制，支持多头层归一化。
    """
    def __init__(self, dim, n_heads, mh_qknorm=False, dtype=None, device=None, operations=None):
        """
        初始化 SingleAttention 模块。

        参数:
            dim (int): 注意力机制的维度。
            n_heads (int): 头的数量。
            mh_qknorm (bool, 可选): 是否使用多头层归一化，默认为 False。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
            operations (module, 可选): 包含自定义操作的模块，默认为 None。
        """
        super().__init__()

        # 设置头的数量
        self.n_heads = n_heads
        # 计算每个头的维度
        self.head_dim = dim // n_heads

        # 定义查询、键、值和输出线性层
        self.w1q = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w1k = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w1v = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w1o = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)

        # 定义查询和键的多头层归一化或层归一化
        self.q_norm1 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim), dtype=dtype, device=device)
            if mh_qknorm
            else operations.LayerNorm(self.head_dim, elementwise_affine=False, dtype=dtype, device=device)
        )
        self.k_norm1 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim), dtype=dtype, device=device)
            if mh_qknorm
            else operations.LayerNorm(self.head_dim, elementwise_affine=False, dtype=dtype, device=device)
        )

    # 使用 torch.compile 装饰器（如果需要，可以取消注释）
    # @torch.compile()
    def forward(self, c):
        """
        前向传播函数，应用单头自注意力机制。

        参数:
            c (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, dim)。

        返回:
            torch.Tensor: 输出张量。
        """
        # 获取输入张量的形状
        bsz, seqlen1, _ = c.shape

        # 计算查询、键和值
        q, k, v = self.w1q(c), self.w1k(c), self.w1v(c)

        # 重塑查询、键和值以适应多头机制
        q = q.view(bsz, seqlen1, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen1, self.n_heads, self.head_dim)
        v = v.view(bsz, seqlen1, self.n_heads, self.head_dim)
        # 应用多头层归一化
        q, k = self.q_norm1(q), self.k_norm1(k)

        # 应用优化的注意力机制
        output = optimized_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), self.n_heads, skip_reshape=True)
        # 应用输出线性层
        c = self.w1o(output)
        return c


class DoubleAttention(nn.Module):
    """
    DoubleAttention 类，实现双头自注意力机制。
    该类同时处理两种不同的输入（c 和 x），并分别计算它们的自注意力。
    """
    def __init__(self, dim, n_heads, mh_qknorm=False, dtype=None, device=None, operations=None):
        """
        初始化 DoubleAttention 模块。

        参数:
            dim (int): 注意力机制的维度。
            n_heads (int): 头的数量。
            mh_qknorm (bool, 可选): 是否使用多头层归一化，默认为 False。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
            operations (module, 可选): 包含自定义操作的模块，默认为 None。
        """
        super().__init__()

        # 设置头的数量
        self.n_heads = n_heads
        # 计算每个头的维度
        self.head_dim = dim // n_heads

        # 定义针对输入 c 的查询、键、值和输出线性层
        self.w1q = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w1k = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w1v = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w1o = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)

        # 定义针对输入 x 的查询、键、值和输出线性层
        self.w2q = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w2k = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w2v = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w2o = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)

        # 定义针对输入 c 的查询和键的多头层归一化或层归一化
        self.q_norm1 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim), dtype=dtype, device=device)
            if mh_qknorm
            else operations.LayerNorm(self.head_dim, elementwise_affine=False, dtype=dtype, device=device)
        )
        self.k_norm1 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim), dtype=dtype, device=device)
            if mh_qknorm
            else operations.LayerNorm(self.head_dim, elementwise_affine=False, dtype=dtype, device=device)
        )

        # 定义针对输入 x 的查询和键的多头层归一化或层归一化
        self.q_norm2 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim), dtype=dtype, device=device)
            if mh_qknorm
            else operations.LayerNorm(self.head_dim, elementwise_affine=False, dtype=dtype, device=device)
        )
        self.k_norm2 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim), dtype=dtype, device=device)
            if mh_qknorm
            else operations.LayerNorm(self.head_dim, elementwise_affine=False, dtype=dtype, device=device)
        )

    # 使用 torch.compile 装饰器（如果需要，可以取消注释）
    # @torch.compile()
    def forward(self, c, x):
        """
        前向传播函数，应用双头自注意力机制。

        参数:
            c (torch.Tensor): 第一种输入张量，形状为 (batch_size, seq_len1, dim)。
            x (torch.Tensor): 第二种输入张量，形状为 (batch_size, seq_len2, dim)。

        返回:
            tuple: 包含处理后的 c 和 x 的元组。
        """
        # 获取输入 c 的形状
        bsz, seqlen1, _ = c.shape
        # 获取输入 x 的形状
        bsz, seqlen2, _ = x.shape

        # 计算针对输入 c 的查询、键和值
        cq, ck, cv = self.w1q(c), self.w1k(c), self.w1v(c)
        # 重塑查询张量
        cq = cq.view(bsz, seqlen1, self.n_heads, self.head_dim)
        # 重塑键张量
        ck = ck.view(bsz, seqlen1, self.n_heads, self.head_dim)
        # 重塑值张量
        cv = cv.view(bsz, seqlen1, self.n_heads, self.head_dim)
        # 应用多头层归一化
        cq, ck = self.q_norm1(cq), self.k_norm1(ck)

        # 计算针对输入 x 的查询、键和值
        xq, xk, xv = self.w2q(x), self.w2k(x), self.w2v(x)
        # 重塑查询张量
        xq = xq.view(bsz, seqlen2, self.n_heads, self.head_dim)
        # 重塑键张量
        xk = xk.view(bsz, seqlen2, self.n_heads, self.head_dim)
        # 重塑值张量
        xv = xv.view(bsz, seqlen2, self.n_heads, self.head_dim)
        # 应用多头层归一化
        xq, xk = self.q_norm2(xq), self.k_norm2(xk)

        # 将查询、键和值连接起来
        q, k, v = (
            torch.cat([cq, xq], dim=1),  # 连接查询张量
            torch.cat([ck, xk], dim=1),  # 连接键张量
            torch.cat([cv, xv], dim=1),  # 连接值张量
        )

        # 应用优化的注意力机制
        output = optimized_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), self.n_heads, skip_reshape=True)

        # 分离处理后的 c 和 x
        c, x = output.split([seqlen1, seqlen2], dim=1)
        c = self.w1o(c)  # 应用输出线性层到 c
        x = self.w2o(x)  # 应用输出线性层到 x

        # 返回处理后的 c 和 x
        return c, x


class MMDiTBlock(nn.Module):
    """
    MMDiTBlock 类，实现混合模态 Transformer 块（MMDiT）。
    该块处理两种输入（c 和 x），并应用多头自注意力机制和前馈神经网络。
    """
    def __init__(self, dim, heads=8, global_conddim=1024, is_last=False, dtype=None, device=None, operations=None):
        """
        初始化 MMDiTBlock 模块。

        参数:
            dim (int): 块的维度。
            heads (int, 可选): 头的数量，默认为 8。
            global_conddim (int, 可选): 全局条件特征的维度，默认为 1024。
            is_last (bool, 可选): 是否为最后一个块，默认为 False。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
            operations (module, 可选): 包含自定义操作的模块，默认为 None。
        """
        super().__init__()

        # 对输入 c 进行层归一化
        self.normC1 = operations.LayerNorm(dim, elementwise_affine=False, dtype=dtype, device=device)
        # 对输出 c 进行层归一化
        self.normC2 = operations.LayerNorm(dim, elementwise_affine=False, dtype=dtype, device=device)
        if not is_last:
            # 如果不是最后一个块，则定义前馈神经网络和调制模块
            self.mlpC = MLP(dim, hidden_dim=dim * 4, dtype=dtype, device=device, operations=operations)
            self.modC = nn.Sequential(
                nn.SiLU(),
                operations.Linear(global_conddim, 6 * dim, bias=False, dtype=dtype, device=device), # 线性层
            )
        else:
            # 如果是最后一个块，则只定义调制模块
            self.modC = nn.Sequential(
                nn.SiLU(),
                operations.Linear(global_conddim, 2 * dim, bias=False, dtype=dtype, device=device), # 线性层
            )

        # 对输入 x 进行层归一化
        self.normX1 = operations.LayerNorm(dim, elementwise_affine=False, dtype=dtype, device=device)
        # 对输出 x 进行层归一化
        self.normX2 = operations.LayerNorm(dim, elementwise_affine=False, dtype=dtype, device=device)
        # 定义前馈神经网络
        self.mlpX = MLP(dim, hidden_dim=dim * 4, dtype=dtype, device=device, operations=operations)
        self.modX = nn.Sequential(
            nn.SiLU(),
            operations.Linear(global_conddim, 6 * dim, bias=False, dtype=dtype, device=device), # 线性层
        )

        # 定义双头自注意力机制
        self.attn = DoubleAttention(dim, heads, dtype=dtype, device=device, operations=operations)
        # 设置是否为最后一个块
        self.is_last = is_last

    # 使用 torch.compile 装饰器（如果需要，可以取消注释）
    # @torch.compile()
    def forward(self, c, x, global_cond, **kwargs):
        """
        前向传播函数，应用 MMDiT 块。

        参数:
            c (torch.Tensor): 输入 c，形状为 (batch_size, seq_len, dim)。
            x (torch.Tensor): 输入 x，形状为 (batch_size, seq_len, dim)。
            global_cond (torch.Tensor): 全局条件张量。
            **kwargs: 其他关键字参数。

        返回:
            tuple: 包含处理后的 c 和 x 的元组。
        """
        # 保存输入 c 和 x 作为残差
        cres, xres = c, x

        # 对输入 c 进行调制
        cshift_msa, cscale_msa, cgate_msa, cshift_mlp, cscale_mlp, cgate_mlp = (
            self.modC(global_cond).chunk(6, dim=1)
        )

        # 应用调制
        c = modulate(self.normC1(c), cshift_msa, cscale_msa)

        # 对输入 x 进行调制
        xshift_msa, xscale_msa, xgate_msa, xshift_mlp, xscale_mlp, xgate_mlp = (
            self.modX(global_cond).chunk(6, dim=1)
        )

        # 应用调制
        x = modulate(self.normX1(x), xshift_msa, xscale_msa)

        # 应用双头自注意力机制
        c, x = self.attn(c, x)

        # 应用层归一化、门控和前馈神经网络到输入 c
        c = self.normC2(cres + cgate_msa.unsqueeze(1) * c)
        c = cgate_mlp.unsqueeze(1) * self.mlpC(modulate(c, cshift_mlp, cscale_mlp))
        c = cres + c

        # 应用层归一化、门控和前馈神经网络到输入 x
        x = self.normX2(xres + xgate_msa.unsqueeze(1) * x)
        x = xgate_mlp.unsqueeze(1) * self.mlpX(modulate(x, xshift_mlp, xscale_mlp))
        x = xres + x

        return c, x


class DiTBlock(nn.Module):
    """
    DiTBlock 类，实现了一个仅处理 X（图像特征）的 Transformer 块。
    该块类似于 MMDiTBlock，但仅处理单一模态的数据。
    """
    # like MMDiTBlock, but it only has X
    def __init__(self, dim, heads=8, global_conddim=1024, dtype=None, device=None, operations=None):
        """
        初始化 DiTBlock 模块。

        参数:
            dim (int): 块的维度。
            heads (int, 可选): 头的数量，默认为 8。
            global_conddim (int, 可选): 全局条件特征的维度，默认为 1024。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
            operations (module, 可选): 包含自定义操作的模块，默认为 None。
        """
        super().__init__()

        # 对输入进行层归一化
        self.norm1 = operations.LayerNorm(dim, elementwise_affine=False, dtype=dtype, device=device)
        # 对输出进行层归一化
        self.norm2 = operations.LayerNorm(dim, elementwise_affine=False, dtype=dtype, device=device)

        self.modCX = nn.Sequential(
            nn.SiLU(),  # SiLU 激活函数
            operations.Linear(global_conddim, 6 * dim, bias=False, dtype=dtype, device=device), # 线性层
        )  # 调制模块，用于生成缩放和平移因子

        # 单头自注意力机制
        self.attn = SingleAttention(dim, heads, dtype=dtype, device=device, operations=operations)
        # 前馈神经网络
        self.mlp = MLP(dim, hidden_dim=dim * 4, dtype=dtype, device=device, operations=operations)

    # 使用 torch.compile 装饰器（如果需要，可以取消注释）
    # @torch.compile()
    def forward(self, cx, global_cond, **kwargs):
        """
        前向传播函数，应用 DiTBlock。

        参数:
            cx (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, dim)。
            global_cond (torch.Tensor): 全局条件张量。
            **kwargs: 其他关键字参数。

        返回:
            torch.Tensor: 输出张量。
        """
        # 保存输入作为残差
        cxres = cx
        # 生成调制因子
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.modCX(
            global_cond
        ).chunk(6, dim=1)
        # 应用调制到输入
        cx = modulate(self.norm1(cx), shift_msa, scale_msa)
        # 应用自注意力机制
        cx = self.attn(cx)
        # 应用层归一化，并添加门控后的残差
        cx = self.norm2(cxres + gate_msa.unsqueeze(1) * cx)
        # 应用调制到 MLP 的输出
        mlpout = self.mlp(modulate(cx, shift_mlp, scale_mlp))
        # 应用门控到 MLP 的输出
        cx = gate_mlp.unsqueeze(1) * mlpout

        # 添加残差
        cx = cxres + cx
        
        # 返回输出张量
        return cx


class TimestepEmbedder(nn.Module):
    """
    TimestepEmbedder 类，实现时间步嵌入。
    该类生成时间步的嵌入向量，用于表示时间信息。
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=None, device=None, operations=None):
        """
        初始化 TimestepEmbedder 模块。

        参数:
            hidden_size (int): 隐藏层的大小。
            frequency_embedding_size (int, 可选): 频率嵌入的维度，默认为 256。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
            operations (module, 可选): 包含自定义操作的模块，默认为 None。
        """
        super().__init__()
        # MLP，用于处理频率嵌入
        self.mlp = nn.Sequential(
            operations.Linear(frequency_embedding_size, hidden_size, dtype=dtype, device=device), # 线性层
            nn.SiLU(),  # SiLU 激活函数
            operations.Linear(hidden_size, hidden_size, dtype=dtype, device=device), # 线性层
        )
        # 设置频率嵌入的维度
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        生成时间步的嵌入向量。

        参数:
            t (torch.Tensor): 输入时间步张量。
            dim (int): 嵌入的维度。
            max_period (int, 可选): 最大周期，默认为 10000。

        返回:
            torch.Tensor: 时间步嵌入张量。
        """
        # 计算维度的一半
        half = dim // 2
        # 生成频率
        freqs = 1000 * torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half 
        ).to(t.device)
        # 计算角度
        args = t[:, None] * freqs[None]
        # 计算正弦和余弦，并拼接
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            # 如果维度为奇数，则填充零
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        # 返回嵌入
        return embedding

    # 使用 torch.compile 装饰器（如果需要，可以取消注释）
    # @torch.compile()
    def forward(self, t, dtype):
        """
        前向传播函数，生成时间步嵌入。

        参数:
            t (torch.Tensor): 输入时间步张量。
            dtype (torch.dtype): 张量的数据类型。

        返回:
            torch.Tensor: 时间步嵌入张量。
        """
        # 生成频率嵌入
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(dtype)
        # 应用 MLP
        t_emb = self.mlp(t_freq)
        # 返回嵌入
        return t_emb


class MMDiT(nn.Module):
    """
    MMDiT 类，实现了一个混合模态 Transformer 模型。
    该模型处理图像和条件输入（如文本），并生成图像的预测。
    """
    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        patch_size=2,
        dim=3072,
        n_layers=36,
        n_double_layers=4,
        n_heads=12,
        global_conddim=3072,
        cond_seq_dim=2048,
        max_seq=32 * 32,
        device=None,
        dtype=None,
        operations=None,
    ):
        """
        初始化 MMDiT 模块。

        参数:
            in_channels (int, 可选): 输入通道数，默认为 4。
            out_channels (int, 可选): 输出通道数，默认为 4。
            patch_size (int, 可选): 补丁大小，默认为 2。
            dim (int, 可选): 模型的维度，默认为 3072。
            n_layers (int, 可选): Transformer 层的数量，默认为 36。
            n_double_layers (int, 可选): 双层 Transformer 层的数量，默认为 4。
            n_heads (int, 可选): 自注意力机制中头的数量，默认为 12。
            global_conddim (int, 可选): 全局条件特征的维度，默认为 3072。
            cond_seq_dim (int, 可选): 条件序列的维度，默认为 2048。
            max_seq (int, 可选): 最大序列长度，默认为 32 * 32。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            operations (module, 可选): 包含自定义操作的模块，默认为 None。
        """
        super().__init__()
        # 设置数据类型
        self.dtype = dtype

        # 初始化时间步嵌入器
        self.t_embedder = TimestepEmbedder(global_conddim, dtype=dtype, device=device, operations=operations)
        
        # 初始化条件序列线性层
        self.cond_seq_linear = operations.Linear(
            cond_seq_dim, dim, bias=False, dtype=dtype, device=device
        )  # linear for something like text sequence.
        # 初始化输入图像线性层
        self.init_x_linear = operations.Linear(
            patch_size * patch_size * in_channels, dim, dtype=dtype, device=device
        )  # init linear for patchified image.

        # 初始化位置编码参数
        self.positional_encoding = nn.Parameter(torch.empty(1, max_seq, dim, dtype=dtype, device=device))
        # 初始化注册标记参数
        self.register_tokens = nn.Parameter(torch.empty(1, 8, dim, dtype=dtype, device=device))

        # 初始化双层 Transformer 层列表
        self.double_layers = nn.ModuleList([])
        # 初始化单层 Transformer 层列表
        self.single_layers = nn.ModuleList([])


        for idx in range(n_double_layers):
            # 添加 MMDiTBlock 到双层列表
            self.double_layers.append(
                MMDiTBlock(dim, n_heads, global_conddim, is_last=(idx == n_layers - 1), dtype=dtype, device=device, operations=operations)
            )

        for idx in range(n_double_layers, n_layers):
            # 添加 DiTBlock 到单层列表
            self.single_layers.append(
                DiTBlock(dim, n_heads, global_conddim, dtype=dtype, device=device, operations=operations)
            )

        # 初始化最终线性层
        self.final_linear = operations.Linear(
            dim, patch_size * patch_size * out_channels, bias=False, dtype=dtype, device=device
        )

        # 调制模块
        self.modF = nn.Sequential(
            nn.SiLU(),  # SiLU 激活函数
            operations.Linear(global_conddim, 2 * dim, bias=False, dtype=dtype, device=device), # 线性层
        )

        # 设置输出通道数
        self.out_channels = out_channels
        # 设置补丁大小
        self.patch_size = patch_size
        # 设置双层 Transformer 层的数量
        self.n_double_layers = n_double_layers
        # 设置 Transformer 层的总数量
        self.n_layers = n_layers

        # 计算高度的最大值
        self.h_max = round(max_seq**0.5)
        # 计算宽度的最大值
        self.w_max = round(max_seq**0.5)

    @torch.no_grad()
    def extend_pe(self, init_dim=(16, 16), target_dim=(64, 64)):
        """
        扩展位置编码（PE）以适应更大的图像尺寸。

        参数:
            init_dim (tuple of int, 可选): 初始维度，默认为 (16, 16)。
            target_dim (tuple of int, 可选): 目标维度，默认为 (64, 64)。
        """
        # 获取初始位置编码数据
        pe_data = self.positional_encoding.data.squeeze(0)[: init_dim[0] * init_dim[1]]

        # 重塑为 2D 形式
        pe_as_2d = pe_data.view(init_dim[0], init_dim[1], -1).permute(2, 0, 1)

        # 使用双线性插值扩展位置编码
        pe_as_2d = F.interpolate(
            pe_as_2d.unsqueeze(0), size=target_dim, mode="bilinear"
        )
        # 重塑回原始形状
        pe_new = pe_as_2d.squeeze(0).permute(1, 2, 0).flatten(0, 1)
        # 更新位置编码数据
        self.positional_encoding.data = pe_new.unsqueeze(0).contiguous()
        # 更新高度和宽度的最大值
        self.h_max, self.w_max = target_dim

    def pe_selection_index_based_on_dim(self, h, w):
        """
        根据图像尺寸选择位置编码的索引。

        参数:
            h (int): 图像的高度。
            w (int): 图像的宽度。

        返回:
            torch.Tensor: 位置编码的索引。
        """
        # 计算补丁数量
        h_p, w_p = h // self.patch_size, w // self.patch_size
        # 生成位置编码索引
        original_pe_indexes = torch.arange(self.positional_encoding.shape[1])
        # 重塑为 2D 形式
        original_pe_indexes = original_pe_indexes.view(self.h_max, self.w_max)

        # 计算起始高度索引
        starth =  self.h_max // 2 - h_p // 2
        # 计算结束高度索引
        endh =starth + h_p
        # 计算起始宽度索引
        startw = self.w_max // 2 - w_p // 2
        # 计算结束宽度索引
        endw = startw + w_p
        # 选择位置编码索引
        original_pe_indexes = original_pe_indexes[
            starth:endh, startw:endw
        ]
        # 返回展平后的索引
        return original_pe_indexes.flatten()

    def unpatchify(self, x, h, w):
        """
        将补丁化的图像数据重新组合成原始图像。

        参数:
            x (torch.Tensor): 补丁化的图像数据。
            h (int): 图像的高度。
            w (int): 图像的宽度。

        返回:
            torch.Tensor: 重新组合后的图像。
        """
        # 获取输出通道数
        c = self.out_channels
        # 获取补丁大小
        p = self.patch_size

        # 重塑张量以分离补丁维度
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        # 使用爱因斯坦求和重塑张量
        x = torch.einsum("nhwpqc->nchpwq", x)
        # 重塑回原始图像形状
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        # 返回图像
        return imgs

    def patchify(self, x):
        """
        将图像数据分割成补丁。

        参数:
            x (torch.Tensor): 输入图像数据。

        返回:
            torch.Tensor: 补丁化的图像数据。
        """
        # 获取输入张量的形状
        B, C, H, W = x.size()
        # 填充图像以适应补丁大小
        x = common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
        # 重塑张量以分割成补丁
        x = x.view(
            B,
            C,
            (H + 1) // self.patch_size,
            self.patch_size,
            (W + 1) // self.patch_size,
            self.patch_size,
        )
        # 重新排列维度并展平补丁维度
        x = x.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)
        # 返回补丁化的图像
        return x

    def apply_pos_embeds(self, x, h, w):
        """
        应用位置嵌入到输入张量。

        参数:
            x (torch.Tensor): 输入张量。
            h (int): 图像的高度。
            w (int): 图像的宽度。

        返回:
            torch.Tensor: 添加了位置嵌入后的张量。
        """
        # 计算补丁化后的高度
        h = (h + 1) // self.patch_size
        # 计算补丁化后的宽度
        w = (w + 1) // self.patch_size
        # 计算最大维度
        max_dim = max(h, w)

        # 获取当前的最大维度
        cur_dim = self.h_max
        # 重塑位置编码并转换为与输入相同的设备和数据类型
        pos_encoding = ops.cast_to_input(self.positional_encoding.reshape(1, cur_dim, cur_dim, -1), x)

        if max_dim > cur_dim:
            # 使用双线性插值扩展位置编码
            pos_encoding = F.interpolate(pos_encoding.movedim(-1, 1), (max_dim, max_dim), mode="bilinear").movedim(1, -1)
            # 更新当前的最大维度
            cur_dim = max_dim

        # 计算起始高度索引
        from_h = (cur_dim - h) // 2
        # 计算起始宽度索引
        from_w = (cur_dim - w) // 2
        # 选择位置编码
        pos_encoding = pos_encoding[:,from_h:from_h+h,from_w:from_w+w]
        # 添加位置嵌入并返回
        return x + pos_encoding.reshape(1, -1, self.positional_encoding.shape[-1])

    def forward(self, x, timestep, context, transformer_options={}, **kwargs):
        """
        前向传播函数，应用 DiTTransformer。

        参数:
            x (torch.Tensor): 输入图像数据。
            timestep (torch.Tensor): 时间步张量。
            context (torch.Tensor): 上下文张量。
            transformer_options (dict, 可选): Transformer 选项，默认为 {}。
            **kwargs: 其他关键字参数。

        返回:
            torch.Tensor: 输出图像数据。
        """
        # 获取 Transformer 选项中的 "patches_replace" 部分
        patches_replace = transformer_options.get("patches_replace", {})
        # patchify x, add PE
        b, c, h, w = x.shape  # 获取输入张量的形状

        # pe_indexes = self.pe_selection_index_based_on_dim(h, w)
        # print(pe_indexes, pe_indexes.shape)

        # 应用输入线性层并补丁化
        x = self.init_x_linear(self.patchify(x))  # B, T_x, D
        # 应用位置嵌入
        x = self.apply_pos_embeds(x, h, w)
        # x = x + self.positional_encoding[:, : x.size(1)].to(device=x.device, dtype=x.dtype)
        # x = x + self.positional_encoding[:, pe_indexes].to(device=x.device, dtype=x.dtype)

        # 处理条件输入
        c_seq = context  # B, T_c, D_c  # 获取上下文张量
        t = timestep  # 获取时间步张量

        # 应用线性层到上下文张量
        c = self.cond_seq_linear(c_seq)  # B, T_c, D
        # 连接注册标记和上下文张量
        c = torch.cat([ops.cast_to_input(self.register_tokens, c).repeat(c.size(0), 1, 1), c], dim=1)

        # 生成全局条件嵌入
        global_cond = self.t_embedder(t, x.dtype)  # B, D

        # 获取 Transformer 选项中的 "dit" 部分
        blocks_replace = patches_replace.get("dit", {})
        if len(self.double_layers) > 0:
            for i, layer in enumerate(self.double_layers):
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["txt"], out["img"] = layer(args["txt"],
                                                       args["img"],
                                                       args["vec"])
                        return out
                    # 应用双层 Transformer 块
                    out = blocks_replace[("double_block", i)]({"img": x, "txt": c, "vec": global_cond}, {"original_block": block_wrap})
                    c = out["txt"]
                    x = out["img"]
                else:
                    # 否则，正常应用双层 Transformer 块
                    c, x = layer(c, x, global_cond, **kwargs)

        if len(self.single_layers) > 0:
            c_len = c.size(1)
            # 连接上下文和图像张量
            cx = torch.cat([c, x], dim=1)
            for i, layer in enumerate(self.single_layers):
                if ("single_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = layer(args["img"], args["vec"])
                        return out
                    
                    # 应用单层 Transformer 块
                    out = blocks_replace[("single_block", i)]({"img": cx, "vec": global_cond}, {"original_block": block_wrap})
                    cx = out["img"]
                else:
                    # 否则，正常应用单层 Transformer 块
                    cx = layer(cx, global_cond, **kwargs)

            # 获取图像部分的输出
            x = cx[:, c_len:]

        # 生成调制因子
        fshift, fscale = self.modF(global_cond).chunk(2, dim=1)

        # 应用调制
        x = modulate(x, fshift, fscale)
        # 应用最终线性层
        x = self.final_linear(x)
        # 重新组合补丁并裁剪到原始图像尺寸
        x = self.unpatchify(x, (h + 1) // self.patch_size, (w + 1) // self.patch_size)[:,:,:h,:w]
        # 返回输出图像张量
        return x
