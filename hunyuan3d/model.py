import torch
from torch import nn
from flux.layers import (
    DoubleStreamBlock,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)


class Hunyuan3Dv2(nn.Module):
    """
    Hunyuan3Dv2 是一个用于3D图像生成或处理的神经网络模型。
    该模型结合了多头自注意力机制、MLP层以及双流和单流块，以实现复杂的特征提取和变换。
    """
    def __init__(
        self,
        in_channels=64,
        context_in_dim=1536,
        hidden_size=1024,
        mlp_ratio=4.0,
        num_heads=16,
        depth=16,
        depth_single_blocks=32,
        qkv_bias=True,
        guidance_embed=False,
        image_model=None,
        dtype=None,
        device=None,
        operations=None
    ):
        """
        初始化 Hunyuan3Dv2 模型。

        参数:
            in_channels (int): 输入图像的通道数，默认为64。
            context_in_dim (int): 条件输入的维度，默认为1536。
            hidden_size (int): 隐藏层的大小，默认为1024。
            mlp_ratio (float): MLP层的扩展比例，默认为4.0。
            num_heads (int): 多头自注意力机制的头数，默认为16。
            depth (int): 双流块的数量，默认为16。
            depth_single_blocks (int): 单流块的数量，默认为32。
            qkv_bias (bool): 在QKV线性变换中是否使用偏置，默认为True。
            guidance_embed (bool): 是否使用引导嵌入，默认为False。
            image_model (nn.Module, optional): 可选的图像模型，默认为None。
            dtype (torch.dtype, optional): 数据类型，默认为None（使用默认类型）。
            device (torch.device, optional): 设备（CPU或GPU），默认为None（自动选择）。
            operations (object, optional): 自定义操作对象，默认为None。
        """
        super().__init__()
        # 设置数据类型
        self.dtype = dtype

        # 检查隐藏层大小是否可以被头数整除
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )

        # 最大周期参数，原始实现中可能存在误解
        self.max_period = 1000  # While reimplementing the model I noticed that they messed up. This 1000 value was meant to be the time_factor but they set the max_period instead
        
        # 线性层用于将输入通道数映射到隐藏层大小
        self.latent_in = operations.Linear(in_channels, hidden_size, bias=True, dtype=dtype, device=device)
        
        # MLP嵌入器，用于将时间步嵌入到隐藏层大小
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=hidden_size, dtype=dtype, device=device, operations=operations)
        
        # 如果使用引导嵌入，则使用另一个 MLP 嵌入器
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=hidden_size, dtype=dtype, device=device, operations=operations) if guidance_embed else None
        )

        # 条件输入的线性层，将条件输入的维度映射到隐藏层大小
        self.cond_in = operations.Linear(context_in_dim, hidden_size, dtype=dtype, device=device)

        # 双流块列表，包含 depth 个双流块
        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    dtype=dtype, device=device, operations=operations
                )
                for _ in range(depth)
            ]
        )

        # 单流块列表，包含 depth_single_blocks 个单流块
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dtype=dtype, device=device, operations=operations
                )
                for _ in range(depth_single_blocks)
            ]
        )

        # 最后一层，将隐藏层大小映射回输出通道数
        self.final_layer = LastLayer(hidden_size, 1, in_channels, dtype=dtype, device=device, operations=operations)

    def forward(self, x, timestep, context, guidance=None, transformer_options={}, **kwargs):
        """
        前向传播过程。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, ..., in_channels)
                注意：输入张量的维度顺序可能需要调整。
            timestep (torch.Tensor): 时间步张量，形状为 (batch_size,)。
            context (torch.Tensor): 条件输入张量，形状为 (batch_size, context_in_dim)。
            guidance (torch.Tensor, optional): 引导输入张量，形状为 (batch_size, 256)，默认为None。
            transformer_options (dict): 包含用于替换Transformer块的可选参数，默认为空字典。
            **kwargs: 其他可选的关键字参数。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, in_channels, ...)。
        """
        # 将输入张量从 (batch_size, ..., in_channels) 转换为 (batch_size, in_channels, ...)
        # 这一步是为了适应后续的线性层处理
        x = x.movedim(-1, -2)  # 将最后一个维度移动到倒数第二个位置

        # 将时间步嵌入到隐藏层大小
        timestep = 1.0 - timestep  # 对时间步进行某种形式的归一化或变换
        txt = context  # 将条件输入赋值给 txt 变量

        # 将输入张量通过 latent_in 线性层映射到隐藏层大小
        img = self.latent_in(x)

        # 对时间步进行时间嵌入
        # timestep_embedding 是一个假设的函数，用于将时间步嵌入到指定维度
        # 这里假设它将时间步嵌入到 256 维，并乘以 max_period
        vec = self.time_in(timestep_embedding(timestep, 256, self.max_period).to(dtype=img.dtype))

        # 如果使用引导嵌入，则对 guidance 进行时间嵌入并添加到 vec 中
        if self.guidance_in is not None:
            if guidance is not None:
                # 将引导嵌入添加到 vec 中
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256, self.max_period).to(img.dtype))

        # 将条件输入通过 cond_in 线性层映射到隐藏层大小
        txt = self.cond_in(txt)

        # 初始化位置编码和注意力掩码为 None
        pe = None
        attn_mask = None

        # 处理 transformer_options 中的替换块
        patches_replace = transformer_options.get("patches_replace", {})  # 获取 "patches_replace" 键对应的字典
        # 从 "patches_replace" 字典中获取 "dit" 键对应的字典
        blocks_replace = patches_replace.get("dit", {})

        # 遍历所有双流块，并根据 transformer_options 进行替换或正常处理
        for i, block in enumerate(self.double_blocks):
            if ("double_block", i) in blocks_replace:
                # 如果当前块需要被替换，则定义一个包装函数
                def block_wrap(args):
                    out = {}
                    # 调用被替换的原始块，并获取输出
                    out["img"], out["txt"] = block(img=args["img"],
                                                   txt=args["txt"],
                                                   vec=args["vec"],
                                                   pe=args["pe"],
                                                   attn_mask=args.get("attn_mask"))
                    return out

                # 使用替换的块进行处理
                out = blocks_replace[("double_block", i)]({"img": img,
                                                           "txt": txt,
                                                           "vec": vec,
                                                           "pe": pe,
                                                           "attn_mask": attn_mask},
                                                          {"original_block": block_wrap})
                # 更新 txt 和 img
                txt = out["txt"]
                img = out["img"]
            else:
                # 如果不需要替换，则正常调用原始块
                img, txt = block(img=img,
                                 txt=txt,
                                 vec=vec,
                                 pe=pe,
                                 attn_mask=attn_mask)
        
        # 将 txt 和 img 在通道维度上进行拼接
        img = torch.cat((txt, img), 1)

        # 遍历所有单流块，并根据 transformer_options 进行替换或正常处理
        for i, block in enumerate(self.single_blocks):
            if ("single_block", i) in blocks_replace:
                # 如果当前块需要被替换，则定义一个包装函数
                def block_wrap(args):
                    out = {}
                    # 调用被替换的原始块，并获取输出
                    out["img"] = block(args["img"],
                                       vec=args["vec"],
                                       pe=args["pe"],
                                       attn_mask=args.get("attn_mask"))
                    return out

                # 使用替换的块进行处理
                out = blocks_replace[("single_block", i)]({"img": img,
                                                           "vec": vec,
                                                           "pe": pe,
                                                           "attn_mask": attn_mask},
                                                          {"original_block": block_wrap})
                # 更新 img
                img = out["img"]
            else:
                # 如果不需要替换，则正常调用原始块
                img = block(img, vec=vec, pe=pe, attn_mask=attn_mask)

        # 从拼接后的 img 中提取 img 部分，假设 txt 在前，img 在后
        img = img[:, txt.shape[1]:, ...]

        # 通过最后一层进行最终的映射，并结合 vec 进行处理
        img = self.final_layer(img, vec)

        # 将最后一个维度与倒数第二个维度交换，并乘以 -1.0
        return img.movedim(-2, -1) * (-1.0)
