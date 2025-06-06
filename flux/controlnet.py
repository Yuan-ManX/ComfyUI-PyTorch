#Original code can be found on: https://github.com/XLabs-AI/x-flux/blob/main/src/flux/controlnet.py
#modified to support different types of flux controlnets

import torch
import math
from torch import Tensor, nn
from einops import rearrange, repeat

from layers import timestep_embedding
from model import Flux
import common_dit


class MistolineCondDownsamplBlock(nn.Module):
    """
    Mistoline 条件下采样模块。
    该模块通过一系列卷积层和激活函数实现下采样，同时应用条件机制。
    """
    def __init__(self, dtype=None, device=None, operations=None):
        """
        初始化 MistolineCondDownsamplBlock 模块。

        参数:
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
            operations (module, 可选): 包含自定义操作的模块，如 Conv2d 和 Linear，默认为 None。
        """
        super().__init__()
        # 定义编码器部分，使用 nn.Sequential 组合一系列操作
        self.encoder = nn.Sequential(
            # 第一层卷积：输入通道 3，输出通道 16，卷积核大小 3，填充 1
            operations.Conv2d(3, 16, 3, padding=1, dtype=dtype, device=device),
            # SiLU 激活函数
            nn.SiLU(),
            # 第二层卷积：输入通道 16，输出通道 16，卷积核大小 1
            operations.Conv2d(16, 16, 1, dtype=dtype, device=device),
            nn.SiLU(),
            # 第三层卷积：输入通道 16，输出通道 16，卷积核大小 3，填充 1
            operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device),
            nn.SiLU(),
            # 第四层卷积：输入通道 16，输出通道 16，卷积核大小 3，填充 1，步幅 2，实现下采样
            operations.Conv2d(16, 16, 3, padding=1, stride=2, dtype=dtype, device=device),
            nn.SiLU(),
            # 第五层卷积：输入通道 16，输出通道 16，卷积核大小 3，填充 1
            operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device),
            nn.SiLU(),
            # 第六层卷积：输入通道 16，输出通道 16，卷积核大小 3，填充 1，步幅 2，实现下采样
            operations.Conv2d(16, 16, 3, padding=1, stride=2, dtype=dtype, device=device),
            nn.SiLU(),
            # 第七层卷积：输入通道 16，输出通道 16，卷积核大小 3，填充 1
            operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Conv2d(16, 16, 3, padding=1, stride=2, dtype=dtype, device=device),
            nn.SiLU(),
            # 第八层卷积：输入通道 16，输出通道 16，卷积核大小 1
            operations.Conv2d(16, 16, 1, dtype=dtype, device=device),
            nn.SiLU(),
            # 第九层卷积：输入通道 16，输出通道 16，卷积核大小 3，填充 1
            operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device)
        )

    def forward(self, x):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, 3, height, width)。

        返回:
            torch.Tensor: 输出张量，经过编码器处理后的结果。
        """
        return self.encoder(x)


class MistolineControlnetBlock(nn.Module):
    """
    Mistoline 控制网络模块。
    该模块通过线性层和激活函数处理输入特征，应用控制机制。
    """
    def __init__(self, hidden_size, dtype=None, device=None, operations=None):
        """
        初始化 MistolineControlnetBlock 模块。

        参数:
            hidden_size (int): 隐藏层的大小，决定了线性层的输入和输出维度。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
            operations (module, 可选): 包含自定义操作的模块，如 Conv2d 和 Linear，默认为 None。
        """
        super().__init__()
        # 定义线性层，将输入特征映射到隐藏层大小
        self.linear = operations.Linear(hidden_size, hidden_size, dtype=dtype, device=device)
        # 定义 SiLU 激活函数
        self.act = nn.SiLU()

    def forward(self, x):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, hidden_size)。

        返回:
            torch.Tensor: 输出张量，经过线性层和激活函数处理后的结果。
        """
        return self.act(self.linear(x))


class ControlNetFlux(Flux):
    """
    ControlNetFlux 类，继承自 Flux 类，扩展了控制网络（ControlNet）的功能。
    该类实现了在模型中加入 ControlNet 模块，以增强对输入图像的控制能力。
    """
    def __init__(self, latent_input=False, num_union_modes=0, mistoline=False, control_latent_channels=None, image_model=None, dtype=None, device=None, operations=None, **kwargs):
        """
        初始化 ControlNetFlux 模块。

        参数:
            latent_input (bool, 可选): 是否使用潜在输入，默认为 False。
            num_union_modes (int, 可选): 联合模式的数量，默认为 0。
            mistoline (bool, 可选): 是否使用 Mistoline 控制块，默认为 False。
            control_latent_channels (int, 可选): 控制潜在通道的数量，默认为 None。
            image_model (nn.Module, 可选): 图像模型，默认为 None。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
            operations (module, 可选): 包含自定义操作的模块，如 Linear, Conv2d, Embedding，默认为 None。
            **kwargs: 其他关键字参数，传递给父类初始化函数。
        """
        # 调用父类 Flux 的初始化方法
        super().__init__(final_layer=False, dtype=dtype, device=device, operations=operations, **kwargs)

        # 定义模型参数
        self.main_model_double = 19  # 主模型中双层块的数量
        self.main_model_single = 38  # 主模型中单层块的数量

        # 是否使用 Mistoline 控制块
        self.mistoline = mistoline

        # add ControlNet blocks
        # 根据是否使用 Mistoline 定义控制块
        if self.mistoline:
            # 使用 MistolineControlnetBlock 作为控制块
            control_block = lambda : MistolineControlnetBlock(self.hidden_size, dtype=dtype, device=device, operations=operations)
        else:
            # 否则，使用普通的线性层作为控制块
            control_block = lambda : operations.Linear(self.hidden_size, self.hidden_size, dtype=dtype, device=device)

        # 初始化控制网络的双层块列表
        self.controlnet_blocks = nn.ModuleList([])
        for _ in range(self.params.depth):
            self.controlnet_blocks.append(control_block())

        # 初始化控制网络的单层块列表
        self.controlnet_single_blocks = nn.ModuleList([])
        for _ in range(self.params.depth_single_blocks):
            self.controlnet_single_blocks.append(control_block())

        # 设置联合模式的数量
        self.num_union_modes = num_union_modes
        self.controlnet_mode_embedder = None
        # 如果联合模式数量大于 0，则初始化控制网络模式嵌入器
        if self.num_union_modes > 0:
            self.controlnet_mode_embedder = operations.Embedding(self.num_union_modes, self.hidden_size, dtype=dtype, device=device)

        # 设置梯度检查点（默认为 False）
        self.gradient_checkpointing = False
        # 设置是否使用潜在输入
        self.latent_input = latent_input
        # 如果未指定控制潜在通道数量，则默认为输入通道数量；否则，乘以 2x2（假设是补丁大小）
        if control_latent_channels is None:
            control_latent_channels = self.in_channels
        else:
            # 假设是补丁大小
            control_latent_channels *= 2 * 2 #patch size

        # 定义位置嵌入输入线性层
        self.pos_embed_input = operations.Linear(control_latent_channels, self.hidden_size, bias=True, dtype=dtype, device=device)
        # 如果不使用潜在输入，则根据是否使用 Mistoline 定义输入提示块
        if not self.latent_input:
            if self.mistoline:
                self.input_cond_block = MistolineCondDownsamplBlock(dtype=dtype, device=device, operations=operations)
            else:
                # 使用一系列卷积层和激活函数作为输入提示块
                self.input_hint_block = nn.Sequential(
                    operations.Conv2d(3, 16, 3, padding=1, dtype=dtype, device=device),
                    nn.SiLU(),
                    operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device),
                    nn.SiLU(),
                    operations.Conv2d(16, 16, 3, padding=1, stride=2, dtype=dtype, device=device),
                    nn.SiLU(),
                    operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device),
                    nn.SiLU(),
                    operations.Conv2d(16, 16, 3, padding=1, stride=2, dtype=dtype, device=device),
                    nn.SiLU(),
                    operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device),
                    nn.SiLU(),
                    operations.Conv2d(16, 16, 3, padding=1, stride=2, dtype=dtype, device=device),
                    nn.SiLU(),
                    operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device)
                )

    def forward_orig(
        self,
        img: Tensor,
        img_ids: Tensor,
        controlnet_cond: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        control_type: Tensor = None,
    ) -> Tensor:
        """
        原始前向传播函数，处理输入图像、文本、时间步等信息，并生成输出。

        参数:
            img (torch.Tensor): 输入图像张量，形状为 (batch_size, channels, height, width)。
            img_ids (torch.Tensor): 图像标识符张量，形状为 (batch_size, ..., 3)。
            controlnet_cond (torch.Tensor): 控制网络条件张量。
            txt (torch.Tensor): 输入文本张量，形状为 (batch_size, ..., 3)。
            txt_ids (torch.Tensor): 文本标识符张量，形状为 (batch_size, ..., 3)。
            timesteps (torch.Tensor): 时间步张量。
            y (torch.Tensor, 可选): 其他输入向量，默认为 None。
            guidance (torch.Tensor, 可选): 指导张量，默认为 None。
            control_type (torch.Tensor, 可选): 控制类型张量，默认为 None。

        返回:
            torch.Tensor: 输出张量。
        """
        # 检查输入张量的维度
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # 如果 y 为 None，则初始化为零
        if y is None:
            y = torch.zeros((img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype)

        # 处理输入图像
        img = self.img_in(img)

        # 处理控制网络条件
        controlnet_cond = self.pos_embed_input(controlnet_cond)
        img = img + controlnet_cond
        # 处理时间步嵌入
        vec = self.time_in(timestep_embedding(timesteps, 256))
        # 如果启用了指导嵌入，则添加指导信息
        if self.params.guidance_embed:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        # 添加向量输入
        vec = vec + self.vector_in(y)
        # 处理文本输入
        txt = self.txt_in(txt)

        # 如果存在控制网络模式嵌入器且控制类型长度大于 0，则嵌入控制类型并将其与文本连接
        if self.controlnet_mode_embedder is not None and len(control_type) > 0:
            control_cond = self.controlnet_mode_embedder(torch.tensor(control_type, device=img.device), out_dtype=img.dtype).unsqueeze(0).repeat((txt.shape[0], 1, 1))
            txt = torch.cat([control_cond, txt], dim=1)
            txt_ids = torch.cat([txt_ids[:,:1], txt_ids], dim=1)

        # 连接文本和图像标识符
        ids = torch.cat((txt_ids, img_ids), dim=1)
        # 获取位置嵌入
        pe = self.pe_embedder(ids)

        # 初始化控制网络双层输出
        controlnet_double = ()

        # 处理双层控制网络块
        for i in range(len(self.double_blocks)):
            img, txt = self.double_blocks[i](img=img, txt=txt, vec=vec, pe=pe)
            controlnet_double = controlnet_double + (self.controlnet_blocks[i](img),)

        # 将文本和图像连接起来
        img = torch.cat((txt, img), 1)

        # 初始化控制网络单层输出
        controlnet_single = ()

        # 处理单层控制网络块
        for i in range(len(self.single_blocks)):
            img = self.single_blocks[i](img, vec=vec, pe=pe)
            controlnet_single = controlnet_single + (self.controlnet_single_blocks[i](img[:, txt.shape[1] :, ...]),)

        # 根据主模型双层块的数量重复控制网络双层输出
        repeat = math.ceil(self.main_model_double / len(controlnet_double))
        if self.latent_input:
            out_input = ()
            for x in controlnet_double:
                    out_input += (x,) * repeat
        else:
            out_input = (controlnet_double * repeat)

        # 构建输出字典
        out = {"input": out_input[:self.main_model_double]}
        if len(controlnet_single) > 0:
            # 根据主模型单层块的数量重复控制网络单层输出
            repeat = math.ceil(self.main_model_single / len(controlnet_single))
            out_output = ()
            if self.latent_input:
                for x in controlnet_single:
                        out_output += (x,) * repeat
            else:
                out_output = (controlnet_single * repeat)
            out["output"] = out_output[:self.main_model_single]
        return out

    def forward(self, x, timesteps, context, y=None, guidance=None, hint=None, **kwargs):
        """
        重载的前向传播函数，处理输入图像、时间步、上下文等信息，并生成输出。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, channels, height, width)。
            timesteps (torch.Tensor): 时间步张量。
            context (torch.Tensor): 上下文张量。
            y (torch.Tensor, 可选): 其他输入向量，默认为 None。
            guidance (torch.Tensor, 可选): 指导张量，默认为 None。
            hint (torch.Tensor, 可选): 提示张量，默认为 None。
            **kwargs: 其他关键字参数。

        返回:
            dict: 包含输入和输出信息的字典。
        """
        patch_size = 2
        # 如果使用潜在输入，则对提示张量进行填充
        if self.latent_input:
            hint = common_dit.pad_to_patch_size(hint, (patch_size, patch_size))
        # 如果使用 Mistoline 控制块，则对提示张量进行缩放和变换
        elif self.mistoline:
            hint = hint * 2.0 - 1.0
            hint = self.input_cond_block(hint)
        # 否则，使用普通的输入提示块
        else:
            hint = hint * 2.0 - 1.0
            hint = self.input_hint_block(hint)

        # 对提示张量进行重塑以适应补丁大小
        hint = rearrange(hint, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        # 获取输入张量的形状
        bs, c, h, w = x.shape
        # 对输入张量进行填充以适应补丁大小
        x = common_dit.pad_to_patch_size(x, (patch_size, patch_size))
        
        # 对输入图像进行重塑以适应补丁大小
        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        # 计算图像的高度和宽度索引
        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)
        # 初始化图像标识符张量
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        # 设置图像标识符的高度和宽度索引
        img_ids[..., 1] = img_ids[..., 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype)[None, :]
        # 重复图像标识符以匹配批次大小
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        # 初始化文本标识符张量
        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        # 调用原始前向传播函数
        return self.forward_orig(img, img_ids, hint, context, txt_ids, timesteps, y, guidance, control_type=kwargs.get("control_type", []))
