#Original code can be found on: https://github.com/black-forest-labs/flux

from dataclasses import dataclass
import torch
from torch import Tensor, nn
from einops import rearrange, repeat

import common_dit
from layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)


@dataclass
class FluxParams:
    """
    FluxParams 数据类，用于存储 Transformer 模型的各种参数。
    """
    in_channels: int  # 输入通道数
    out_channels: int  # 输出通道数
    vec_in_dim: int  # 向量输入维度
    context_in_dim: int  # 上下文输入维度
    hidden_size: int  # 隐藏层大小
    mlp_ratio: float  # MLP 中隐藏层大小的比例
    num_heads: int  # 自注意力机制中头的数量
    depth: int  # 双流块（DoubleStreamBlock）的深度，即堆叠的层数
    depth_single_blocks: int  # 单流块（SingleStreamBlock）的深度，即堆叠的层数
    axes_dim: list  # 轴的维度列表，用于位置嵌入
    theta: int  # 旋转位置编码（RoPE）的旋转角度参数
    patch_size: int  # 补丁大小，用于将图像分割成小块
    qkv_bias: bool  # 查询、键和值线性层是否使用偏置
    guidance_embed: bool  # 是否使用指导嵌入


class Flux(nn.Module):
    """
    Flux 类，实现了一个用于流匹配的 Transformer 模型。
    该模型处理序列数据，并结合了图像、文本和时间步等信息进行建模。
    """

    def __init__(self, image_model=None, final_layer=True, dtype=None, device=None, operations=None, **kwargs):
        """
        初始化 Flux 模型。

        参数:
            image_model (nn.Module, 可选): 图像模型，默认为 None。
            final_layer (bool, 可选): 是否包含最后一层，默认为 True。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
            operations (module, 可选): 包含自定义操作的模块，默认为 None。
            **kwargs: 其他关键字参数，用于初始化 FluxParams。
        """
        super().__init__()
        self.dtype = dtype
        # 初始化 FluxParams 对象，传入关键字参数
        params = FluxParams(**kwargs)
        self.params = params
        self.patch_size = params.patch_size

        # 计算输入和输出通道数，考虑补丁大小
        self.in_channels = params.in_channels * params.patch_size * params.patch_size
        self.out_channels = params.out_channels * params.patch_size * params.patch_size
        
        # 检查隐藏层大小是否可以被头的数量整除
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        # 计算位置嵌入的维度
        pe_dim = params.hidden_size // params.num_heads

        # 检查轴的维度之和是否等于位置嵌入的维度
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        
        # 设置隐藏层大小
        self.hidden_size = params.hidden_size
        # 设置头的数量
        self.num_heads = params.num_heads
        # 初始化位置嵌入嵌入器
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        # 初始化图像输入线性层，将输入通道数映射到隐藏层大小
        self.img_in = operations.Linear(self.in_channels, self.hidden_size, bias=True, dtype=dtype, device=device)
        # 初始化时间步嵌入的 MLP 嵌入器
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, dtype=dtype, device=device, operations=operations)
        # 初始化向量输入的 MLP 嵌入器
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size, dtype=dtype, device=device, operations=operations)
        # 初始化指导嵌入的 MLP 嵌入器，如果启用了指导嵌入，则使用 MLP；否则，使用恒等函数
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, dtype=dtype, device=device, operations=operations) if params.guidance_embed else nn.Identity()
        )
        # 初始化文本输入线性层，将上下文输入维度映射到隐藏层大小
        self.txt_in = operations.Linear(params.context_in_dim, self.hidden_size, dtype=dtype, device=device)

        # 初始化双流块列表，每个双流块包含隐藏层大小、头的数量、MLP 比例和 QKV 偏置等参数
        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    dtype=dtype, device=device, operations=operations
                )
                for _ in range(params.depth)
            ]
        )

        # 初始化单流块列表，每个单流块包含隐藏层大小、头的数量和 MLP 比例等参数
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio, dtype=dtype, device=device, operations=operations)
                for _ in range(params.depth_single_blocks)
            ]
        )

        # 如果包含最后一层，则初始化最后一层
        if final_layer:
            self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels, dtype=dtype, device=device, operations=operations)

    def forward_orig(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        control = None,
        transformer_options={},
        attn_mask: Tensor = None,
    ) -> Tensor:
        """
        原始前向传播函数，处理输入图像、文本、时间步等信息，并生成输出。

        参数:
            img (torch.Tensor): 输入图像张量，形状为 (batch_size, channels, height, width)。
            img_ids (torch.Tensor): 输入图像的标识符张量，形状为 (batch_size, ..., 3)。
            txt (torch.Tensor): 输入文本张量，形状为 (batch_size, ..., 3)。
            txt_ids (torch.Tensor): 输入文本的标识符张量，形状为 (batch_size, ..., 3)。
            timesteps (torch.Tensor): 输入时间步张量。
            y (torch.Tensor, 可选): 其他输入向量，默认为 None。
            guidance (torch.Tensor, 可选): 指导张量，默认为 None。
            control (dict, 可选): 控制网络信息，默认为 None。
            transformer_options (dict, 可选): Transformer 模型的选项，默认为 {}。
            attn_mask (torch.Tensor, 可选): 注意力掩码，默认为 None。

        返回:
            torch.Tensor: 输出张量。
        """
        # 如果 y 为 None，则初始化为零
        if y is None:
            y = torch.zeros((img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype)

        # 获取 Transformer 选项中的 "patches_replace" 部分
        patches_replace = transformer_options.get("patches_replace", {})
        # 检查输入张量的维度
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # 处理输入图像
        img = self.img_in(img)
        # 生成时间步嵌入，并转换为与 img 相同的数据类型
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        # 如果启用了指导嵌入，则添加指导信息
        if self.params.guidance_embed:
            if guidance is not None:
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        # 添加向量输入
        vec = vec + self.vector_in(y[:,:self.params.vec_in_dim])
        # 处理文本输入
        txt = self.txt_in(txt)

        # 如果提供了图像标识符，则连接文本和图像标识符，并生成位置嵌入
        if img_ids is not None:
            ids = torch.cat((txt_ids, img_ids), dim=1)
            pe = self.pe_embedder(ids)
        else:
            pe = None

        # 获取 Transformer 选项中的 "dit" 部分，用于替换双流块
        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.double_blocks):
            # 如果当前双流块需要被替换，则执行替换操作
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(img=args["img"],
                                                   txt=args["txt"],
                                                   vec=args["vec"],
                                                   pe=args["pe"],
                                                   attn_mask=args.get("attn_mask"))
                    return out

                # 调用替换函数，传入当前图像、文本、向量、位置嵌入和注意力掩码
                out = blocks_replace[("double_block", i)]({"img": img,
                                                           "txt": txt,
                                                           "vec": vec,
                                                           "pe": pe,
                                                           "attn_mask": attn_mask},
                                                          {"original_block": block_wrap})
                # 更新文本和图像张量
                txt = out["txt"]
                img = out["img"]
            else:
                # 否则，正常执行双流块的前向传播
                img, txt = block(img=img,
                                 txt=txt,
                                 vec=vec,
                                 pe=pe,
                                 attn_mask=attn_mask)

            # 如果提供了控制网络信息，则处理控制网络输入
            if control is not None: # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add

        # 将文本和图像张量连接起来
        img = torch.cat((txt, img), 1)

        # 处理单流块
        for i, block in enumerate(self.single_blocks):
            # 如果当前单流块需要被替换，则执行替换操作
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"],
                                       vec=args["vec"],
                                       pe=args["pe"],
                                       attn_mask=args.get("attn_mask"))
                    return out

                # 调用替换函数，传入当前图像、向量、位置嵌入和注意力掩码
                out = blocks_replace[("single_block", i)]({"img": img,
                                                           "vec": vec,
                                                           "pe": pe,
                                                           "attn_mask": attn_mask},
                                                          {"original_block": block_wrap})
                # 更新图像张量
                img = out["img"]
            else:
                # 否则，正常执行单流块的前向传播
                img = block(img, vec=vec, pe=pe, attn_mask=attn_mask)

            # 如果提供了控制网络输出信息，则处理控制网络输出
            if control is not None: # Controlnet
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, txt.shape[1] :, ...] += add

        # 移除文本部分，只保留图像部分
        img = img[:, txt.shape[1] :, ...]

        # 应用最后一层
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img

    def forward(self, x, timestep, context, y=None, guidance=None, control=None, transformer_options={}, **kwargs):
        """
        重载的前向传播函数，处理输入图像、时间步、上下文等信息，并生成输出。

        参数:
            x (torch.Tensor): 输入图像张量，形状为 (batch_size, channels, height, width)。
            timestep (torch.Tensor): 输入时间步张量。
            context (torch.Tensor): 输入文本上下文张量。
            y (torch.Tensor, 可选): 其他输入向量，默认为 None。
            guidance (torch.Tensor, 可选): 指导张量，默认为 None。
            control (dict, 可选): 控制网络信息，默认为 None。
            transformer_options (dict, 可选): Transformer 模型的选项，默认为 {}。
            **kwargs: 其他关键字参数。

        返回:
            torch.Tensor: 输出张量。
        """
        # 获取输入图像的形状
        bs, c, h, w = x.shape
        # 获取补丁大小
        patch_size = self.patch_size
        # 对输入图像进行填充以适应补丁大小
        x = common_dit.pad_to_patch_size(x, (patch_size, patch_size))

        # 重塑图像张量以适应补丁大小
        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        # 计算图像的高度和宽度索引
        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)

        # 初始化图像标识符张量
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)

        # 设置图像标识符的高度和宽度索引
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).unsqueeze(0)
        
        # 重复图像标识符以匹配批次大小
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        # 初始化文本标识符张量
        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        # 调用原始前向传播函数处理输入数据
        out = self.forward_orig(img, img_ids, context, txt_ids, timestep, y, guidance, control, transformer_options, attn_mask=kwargs.get("attention_mask", None))
        # 重塑输出张量以恢复原始图像形状
        return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]
