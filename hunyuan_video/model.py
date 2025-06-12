#Based on Flux code because of weird hunyuan video code license.

import torch
import flux.layers
import modules.diffusionmodules.mmdit
from modules.attention import optimized_attention


from dataclasses import dataclass
from einops import repeat

from torch import Tensor, nn

from flux.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding
)

import common_dit


@dataclass
class HunyuanVideoParams:
    """
    HunyuanVideoParams 数据类，用于存储视频处理模型的各种参数。
    """
    in_channels: int  # 输入通道数
    out_channels: int  # 输出通道数
    vec_in_dim: int  # 向量输入维度
    context_in_dim: int  # 上下文输入维度
    hidden_size: int  # 隐藏层大小
    mlp_ratio: float  # MLP 中隐藏层大小的比例
    num_heads: int  # 自注意力机制中头的数量
    depth: int  # Transformer 块的深度，即堆叠的层数
    depth_single_blocks: int  # 单个 Transformer 块的深度，即堆叠的层数
    axes_dim: list  # 轴的维度列表，用于位置编码
    theta: int  # 旋转位置编码（RoPE）的旋转角度参数
    patch_size: list  # 补丁大小列表，用于将视频分割成小块
    qkv_bias: bool  # 查询、键和值线性层是否使用偏置
    guidance_embed: bool  # 是否使用指导嵌入


class SelfAttentionRef(nn.Module):
    """
    SelfAttentionRef 类，实现了一个自注意力机制。
    该类实现了多头自注意力机制，支持查询、键和值的线性投影。
    """
    def __init__(self, dim: int, qkv_bias: bool = False, dtype=None, device=None, operations=None):
        """
        初始化 SelfAttentionRef 模块。

        参数:
            dim (int): 注意力机制的维度。
            qkv_bias (bool, 可选): 查询、键和值线性层是否使用偏置，默认为 False。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
            operations (module, 可选): 包含自定义操作的模块，默认为 None。
        """
        super().__init__()
        # 定义查询、键和值的线性投影层
        self.qkv = operations.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        # 定义输出投影层
        self.proj = operations.Linear(dim, dim, dtype=dtype, device=device)


class TokenRefinerBlock(nn.Module):
    """
    TokenRefinerBlock 类，实现了一个令牌细化块（Token Refiner Block）。
    该块包含自适应层归一化（adaLN）、自注意力机制和前馈神经网络（MLP）。
    
    参数:
        hidden_size (int): 隐藏层的维度大小。
        heads (int): 自注意力机制中的头数。
        dtype (torch.dtype, optional): 张量的数据类型。
        device (torch.device, optional): 张量所在的设备。
        operations (Module, optional): 包含线性层和层归一化等操作的模块。
    """
    def __init__(
        self,
        hidden_size,
        heads,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()
        self.heads = heads
        # MLP的隐藏层维度为hidden_size的4倍
        mlp_hidden_dim = hidden_size * 4

        # 自适应层归一化（adaLN）调制模块
        # 首先通过SiLU激活函数，然后通过一个线性层将hidden_size映射到2 * hidden_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),  # SiLU激活函数
            operations.Linear(hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device),
        )

        # 第一个层归一化
        self.norm1 = operations.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
        # 自注意力机制模块
        self.self_attn = SelfAttentionRef(hidden_size, True, dtype=dtype, device=device, operations=operations)

        # 第二个层归一化
        self.norm2 = operations.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)

        # 前馈神经网络（MLP）模块
        # 包含两个线性层，中间使用SiLU激活函数
        self.mlp = nn.Sequential(
            operations.Linear(hidden_size, mlp_hidden_dim, bias=True, dtype=dtype, device=device),
            nn.SiLU(),  # SiLU激活函数
            operations.Linear(mlp_hidden_dim, hidden_size, bias=True, dtype=dtype, device=device),
        )

    def forward(self, x, c, mask):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入的令牌嵌入，形状为 (batch_size, seq_length, hidden_size)。
            c (torch.Tensor): 条件向量，形状为 (batch_size, cond_dim)。
            mask (torch.Tensor, optional): 注意力掩码，形状为 (batch_size, seq_length, seq_length)。

        返回:
            torch.Tensor: 细化后的令牌嵌入，形状为 (batch_size, seq_length, hidden_size)。
        """
        # 将条件向量c通过adaLN调制模块，并拆分为两个部分：mod1和mod2
        mod1, mod2 = self.adaLN_modulation(c).chunk(2, dim=1) # mod1和mod2形状均为 (batch_size, hidden_size)

        # 对输入x进行第一个层归一化
        norm_x = self.norm1(x)  # 形状保持不变

        # 通过自注意力机制的前向传播
        qkv = self.self_attn.qkv(norm_x)  # 生成查询、键和值，形状为 (batch_size, seq_length, 3 * hidden_size)

        # 重塑qkv以适应多头注意力机制
        # 形状为 (batch_size, seq_length, 3, heads, hidden_size // heads)
        q, k, v = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, self.heads, -1).permute(2, 0, 3, 1, 4)

        # 计算注意力权重并进行优化注意力计算
        # 形状为 (batch_size, heads, seq_length, hidden_size // heads)
        attn = optimized_attention(q, k, v, self.heads, mask=mask, skip_reshape=True)

        # 将注意力输出投影回原始维度，并进行残差连接
        x = x + self.self_attn.proj(attn) * mod1.unsqueeze(1) # 形状为 (batch_size, seq_length, hidden_size)

        # 通过MLP的前向传播，并进行残差连接
        # 形状为 (batch_size, seq_length, hidden_size)
        x = x + self.mlp(self.norm2(x)) * mod2.unsqueeze(1)
        return x


class IndividualTokenRefiner(nn.Module):
    """
    IndividualTokenRefiner 类，包含多个 TokenRefinerBlock，用于对令牌进行细化。
    
    参数:
        hidden_size (int): 隐藏层的维度大小。
        heads (int): 自注意力机制中的头数。
        num_blocks (int): TokenRefinerBlock 的数量。
        dtype (torch.dtype, optional): 张量的数据类型。
        device (torch.device, optional): 张量所在的设备。
        operations (Module, optional): 包含线性层和层归一化等操作的模块。
    """
    def __init__(
        self,
        hidden_size,
        heads,
        num_blocks,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()
        # 初始化一个包含多个 TokenRefinerBlock 的列表
        self.blocks = nn.ModuleList(
            [
                TokenRefinerBlock(
                    hidden_size=hidden_size,
                    heads=heads,
                    dtype=dtype,
                    device=device,
                    operations=operations
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x, c, mask):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入的令牌嵌入，形状为 (batch_size, seq_length, hidden_size)。
            c (torch.Tensor): 条件向量，形状为 (batch_size, cond_dim)。
            mask (torch.Tensor, optional): 注意力掩码，形状为 (batch_size, seq_length, seq_length)。

        返回:
            torch.Tensor: 细化后的令牌嵌入，形状为 (batch_size, seq_length, hidden_size)。
        """
        # 如果存在掩码，则将其调整为适合多头注意力的形状
        m = None
        if mask is not None:
            # 形状为 (batch_size, 1, seq_length, seq_length)
            m = mask.view(mask.shape[0], 1, 1, mask.shape[1]).repeat(1, 1, mask.shape[1], 1)
            # 转置并相加，得到对称的掩码矩阵
            m = m + m.transpose(2, 3)

        # 依次通过每个 TokenRefinerBlock
        for block in self.blocks:
            x = block(x, c, m)
        return x



class TokenRefiner(nn.Module):
    """
    TokenRefiner 类，整合了输入嵌入器、时间步嵌入器、条件嵌入器和 IndividualTokenRefiner，用于对令牌进行全面的细化。

    参数:
        text_dim (int): 文本嵌入的维度大小。
        hidden_size (int): 隐藏层的维度大小。
        heads (int): 自注意力机制中的头数。
        num_blocks (int): TokenRefinerBlock 的数量。
        dtype (torch.dtype, optional): 张量的数据类型。
        device (torch.device, optional): 张量所在的设备。
        operations (Module, optional): 包含线性层和层归一化等操作的模块。
    """
    def __init__(
        self,
        text_dim,
        hidden_size,
        heads,
        num_blocks,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()

        # 输入嵌入器，将文本嵌入映射到隐藏层维度
        self.input_embedder = operations.Linear(text_dim, hidden_size, bias=True, dtype=dtype, device=device)

        # 时间步嵌入器，使用 MLPEmbedder 将时间步嵌入到 256 维
        self.t_embedder = MLPEmbedder(256, hidden_size, dtype=dtype, device=device, operations=operations)

        # 条件嵌入器，使用 MLPEmbedder 将条件向量嵌入到隐藏层维度
        self.c_embedder = MLPEmbedder(text_dim, hidden_size, dtype=dtype, device=device, operations=operations)
        
        # 令牌细化器，包含多个 TokenRefinerBlock
        self.individual_token_refiner = IndividualTokenRefiner(hidden_size, heads, num_blocks, dtype=dtype, device=device, operations=operations)

    def forward(
        self,
        x,
        timesteps,
        mask,
    ):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入的令牌嵌入，形状为 (batch_size, seq_length, text_dim)。
            timesteps (torch.Tensor): 时间步张量，形状为 (batch_size,)。
            mask (torch.Tensor, optional): 注意力掩码，形状为 (batch_size, seq_length, seq_length)。

        返回:
            torch.Tensor: 细化后的令牌嵌入，形状为 (batch_size, seq_length, hidden_size)。
        """
        # 生成时间步嵌入，并转换为与x相同的数据类型
        # 形状为 (batch_size, hidden_size)
        t = self.t_embedder(timestep_embedding(timesteps, 256, time_factor=1.0).to(x.dtype))

        # 计算条件向量c
        # 如果存在掩码，则计算加权平均；否则，计算平均嵌入
        # TODO: 这里的注释提到当x的长度与token长度相同时有效，否则可能会出错
        # m = mask.float().unsqueeze(-1)
        # c = (x.float() * m).sum(dim=1) / m.sum(dim=1)
        # 计算平均嵌入，形状为 (batch_size, text_dim)
        c = x.sum(dim=1) / x.shape[1]

        # 将时间步嵌入与条件嵌入相加，得到最终的c
        c = t + self.c_embedder(c.to(x.dtype)) # 形状为 (batch_size, hidden_size)

        # 将输入x通过输入嵌入器进行嵌入  
        x = self.input_embedder(x)  # 形状为 (batch_size, seq_length, hidden_size)

        # 通过令牌细化器进行细化
        x = self.individual_token_refiner(x, c, mask) # 形状保持不变
        return x


class HunyuanVideo(nn.Module):
    """
    HunyuanVideo 类是一个用于序列流匹配的 Transformer 模型。
    该模型集成了图像、时间步、向量、文本和指导信息的嵌入与处理，
    并通过双流和单流块进行多层次的特征提取和融合。
    """

    def __init__(self, image_model=None, final_layer=True, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype

        # 解析参数，使用 HunyuanVideoParams 类来组织参数
        params = HunyuanVideoParams(**kwargs)
        self.params = params

        # 从参数中提取关键参数
        self.patch_size = params.patch_size  # 补丁大小
        self.in_channels = params.in_channels  # 输入通道数
        self.out_channels = params.out_channels  # 输出通道数

        # 检查隐藏层大小是否可以被头数整除
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        # 位置嵌入的维度
        pe_dim = params.hidden_size // params.num_heads

        # 检查位置嵌入的维度是否与 axes_dim 的和匹配
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        
        # 初始化关键参数
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads

        # 初始化位置嵌入器，用于嵌入多维位置信息
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)

        # 初始化图像输入嵌入器，将图像分割成小块并嵌入
        self.img_in = modules.diffusionmodules.mmdit.PatchEmbed(None, self.patch_size, self.in_channels, self.hidden_size, conv3d=True, dtype=dtype, device=device, operations=operations)
        
        # 初始化时间步嵌入器，将时间步嵌入到隐藏层维度
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, dtype=dtype, device=device, operations=operations)
        
        # 初始化向量输入嵌入器，将向量嵌入到隐藏层维度
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size, dtype=dtype, device=device, operations=operations)
        
        # 初始化指导信息嵌入器，如果启用指导信息，则使用 MLPEmbedder，否则使用恒等映射
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, dtype=dtype, device=device, operations=operations) if params.guidance_embed else nn.Identity()
        )

        # 初始化文本输入嵌入器，使用 TokenRefiner 对文本进行细化
        self.txt_in = TokenRefiner(params.context_in_dim, self.hidden_size, self.num_heads, 2, dtype=dtype, device=device, operations=operations)

        # 初始化双流块列表，用于同时处理图像和文本
        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    flipped_img_txt=True,
                    dtype=dtype, device=device, operations=operations
                )
                for _ in range(params.depth)
            ]
        )

        # 初始化单流块列表，用于处理单一数据流
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio, dtype=dtype, device=device, operations=operations)
                for _ in range(params.depth_single_blocks)
            ]
        )

        # 如果需要最终层，则初始化最后一层，用于生成最终输出
        if final_layer:
            self.final_layer = LastLayer(self.hidden_size, self.patch_size[-1], self.out_channels, dtype=dtype, device=device, operations=operations)

    def forward_orig(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        txt_mask: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        guiding_frame_index=None,
        ref_latent=None,
        control=None,
        transformer_options={},
    ) -> Tensor:
        """
        原始前向传播方法，处理输入数据并通过模型进行前向传播。

        参数:
            img (torch.Tensor): 输入图像张量，形状为 (batch_size, in_channels, t, h, w)。
            img_ids (torch.Tensor): 图像标识符张量，形状为 (batch_size, t_patch_h patch_w, 3)。
            txt (torch.Tensor): 输入文本张量，形状为 (batch_size, seq_length, text_dim)。
            txt_ids (torch.Tensor): 文本标识符张量，形状为 (batch_size, seq_length, 3)。
            txt_mask (torch.Tensor): 文本掩码张量，形状为 (batch_size, seq_length)。
            timesteps (torch.Tensor): 时间步张量，形状为 (batch_size,)。
            y (torch.Tensor): 目标向量张量，形状为 (batch_size, vec_in_dim)。
            guidance (torch.Tensor, optional): 指导信息张量，形状为 (batch_size, 256)。
            guiding_frame_index (int, optional): 指导帧索引。
            ref_latent (torch.Tensor, optional): 参考潜在向量张量，形状为 (batch_size, in_channels, t, h, w)。
            control (Dict[str, torch.Tensor], optional): 控制信息字典。
            transformer_options (Dict): Transformer 选项。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_channels, t, h, w)。
        """

        # 获取补丁替换选项
        patches_replace = transformer_options.get("patches_replace", {})

        # 记录初始形状
        initial_shape = list(img.shape)
        
        # 将图像通过图像输入嵌入器进行嵌入
        img = self.img_in(img)

        # 生成时间步嵌入
        vec = self.time_in(timestep_embedding(timesteps, 256, time_factor=1.0).to(img.dtype))

        # 如果存在参考潜在向量，则进行相应处理
        if ref_latent is not None:
            # 获取参考潜在向量的图像标识符
            ref_latent_ids = self.img_ids(ref_latent)
            # 嵌入参考潜在向量
            ref_latent = self.img_in(ref_latent)
            # 在最后一个维度上拼接
            img = torch.cat([ref_latent, img], dim=-2)
            # 修改参考潜在向量的标识符
            ref_latent_ids[..., 0] = -1
            # 修改标识符
            ref_latent_ids[..., 2] += (initial_shape[-1] // self.patch_size[-1])
            # 拼接图像标识符
            img_ids = torch.cat([ref_latent_ids, img_ids], dim=-2)

        # 如果存在指导帧索引，则进行相应处理
        if guiding_frame_index is not None:
            token_replace_vec = self.time_in(timestep_embedding(guiding_frame_index, 256, time_factor=1.0))
            vec_ = self.vector_in(y[:, :self.params.vec_in_dim])
            vec = torch.cat([(vec_ + token_replace_vec).unsqueeze(1), (vec_ + vec).unsqueeze(1)], dim=1)
            frame_tokens = (initial_shape[-1] // self.patch_size[-1]) * (initial_shape[-2] // self.patch_size[-2])
            modulation_dims = [(0, frame_tokens, 0), (frame_tokens, None, 1)]
            modulation_dims_txt = [(0, None, 1)]
        else:
            vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])
            modulation_dims = None
            modulation_dims_txt = None

        # 如果嵌入指导信息，则将指导信息嵌入并添加到 vec 中
        if self.params.guidance_embed:
            if guidance is not None:
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        # 处理文本掩码
        if txt_mask is not None and not torch.is_floating_point(txt_mask):
            txt_mask = (txt_mask - 1).to(img.dtype) * torch.finfo(img.dtype).max

        # 将文本通过文本输入嵌入器进行嵌入
        txt = self.txt_in(txt, timesteps, txt_mask)

        # 拼接图像和文本标识符
        ids = torch.cat((img_ids, txt_ids), dim=1)
        pe = self.pe_embedder(ids)

        img_len = img.shape[1]

        # 处理注意力掩码
        if txt_mask is not None:
            attn_mask_len = img_len + txt.shape[1]
            attn_mask = torch.zeros((1, 1, attn_mask_len), dtype=img.dtype, device=img.device)
            attn_mask[:, 0, img_len:] = txt_mask
        else:
            attn_mask = None

        # 处理双流块替换选项
        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.double_blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], attn_mask=args["attention_mask"], modulation_dims_img=args["modulation_dims_img"], modulation_dims_txt=args["modulation_dims_txt"])
                    return out

                out = blocks_replace[("double_block", i)]({"img": img, "txt": txt, "vec": vec, "pe": pe, "attention_mask": attn_mask, 'modulation_dims_img': modulation_dims, 'modulation_dims_txt': modulation_dims_txt}, {"original_block": block_wrap})
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask, modulation_dims_img=modulation_dims, modulation_dims_txt=modulation_dims_txt)

            # 处理 Controlnet 控制信息
            if control is not None: # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add

        # 拼接图像和文本
        img = torch.cat((img, txt), 1)

        # 处理单流块
        for i, block in enumerate(self.single_blocks):
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"], attn_mask=args["attention_mask"], modulation_dims=args["modulation_dims"])
                    return out

                out = blocks_replace[("single_block", i)]({"img": img, "vec": vec, "pe": pe, "attention_mask": attn_mask, 'modulation_dims': modulation_dims}, {"original_block": block_wrap})
                img = out["img"]
            else:
                img = block(img, vec=vec, pe=pe, attn_mask=attn_mask, modulation_dims=modulation_dims)

            # 处理 Controlnet 控制信息
            if control is not None: # Controlnet
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, : img_len] += add

        # 保留原始图像部分
        img = img[:, : img_len]

        # 如果存在参考潜在向量，则进行相应处理
        if ref_latent is not None:
            img = img[:, ref_latent.shape[1]:]

        # 通过最终层生成输出
        img = self.final_layer(img, vec, modulation_dims=modulation_dims)  # (N, T, patch_size ** 2 * out_channels)

        # 重塑输出形状
        shape = initial_shape[-3:]
        for i in range(len(shape)):
            shape[i] = shape[i] // self.patch_size[i]
        img = img.reshape([img.shape[0]] + shape + [self.out_channels] + self.patch_size)
        img = img.permute(0, 4, 1, 5, 2, 6, 3, 7)
        img = img.reshape(initial_shape[0], self.out_channels, initial_shape[2], initial_shape[3], initial_shape[4])
        return img

    def img_ids(self, x):
        """
        生成图像标识符张量，用于标识图像的不同维度。

        参数:
            x (torch.Tensor): 输入图像张量，形状为 (batch_size, in_channels, t, h, w)。

        返回:
            torch.Tensor: 图像标识符张量，形状为 (t_patch_h patch_w, 3)。
        """
        bs, c, t, h, w = x.shape
        patch_size = self.patch_size
        t_len = ((t + (patch_size[0] // 2)) // patch_size[0])
        h_len = ((h + (patch_size[1] // 2)) // patch_size[1])
        w_len = ((w + (patch_size[2] // 2)) // patch_size[2])
        img_ids = torch.zeros((t_len, h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(0, t_len - 1, steps=t_len, device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
        img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).reshape(1, -1, 1)
        img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).reshape(1, 1, -1)
        return repeat(img_ids, "t h w c -> b (t h w) c", b=bs)

    def forward(self, x, timestep, context, y, guidance=None, attention_mask=None, guiding_frame_index=None, ref_latent=None, control=None, transformer_options={}, **kwargs):
        """
        前向传播方法，调用原始前向传播方法并返回输出。

        参数:
            x (torch.Tensor): 输入图像张量，形状为 (batch_size, in_channels, t, h, w)。
            timestep (torch.Tensor): 时间步张量，形状为 (batch_size,)。
            context (torch.Tensor): 上下文张量，形状为 (batch_size, seq_length, text_dim)。
            y (torch.Tensor): 目标向量张量，形状为 (batch_size, vec_in_dim)。
            guidance (torch.Tensor, optional): 指导信息张量，形状为 (batch_size, 256)。
            attention_mask (torch.Tensor, optional): 注意力掩码张量。
            guiding_frame_index (int, optional): 指导帧索引。
            ref_latent (torch.Tensor, optional): 参考潜在向量。
            control (Dict[str, torch.Tensor], optional): 控制信息字典。
            transformer_options (Dict): Transformer 选项。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_channels, t, h, w)。
        """
        bs, c, t, h, w = x.shape
        # 生成图像标识符
        img_ids = self.img_ids(x)
        # 初始化文本标识符
        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        out = self.forward_orig(x, img_ids, context, txt_ids, attention_mask, timestep, y, guidance, guiding_frame_index, ref_latent, control=control, transformer_options=transformer_options)
        return out
