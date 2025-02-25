import torch

from ldm.modules.attention import optimized_attention_for_device
import ops


class CLIPAttention(torch.nn.Module):
    """
    CLIP模型中的多头自注意力机制模块。

    参数:
    - embed_dim (int): 嵌入维度（embedding dimension）。
    - heads (int): 注意力头的数量（number of attention heads）。
    - dtype (torch.dtype): 数据类型，如torch.float32。
    - device (torch.device): 计算设备，如'cuda'或'cpu'。
    - operations (Operations): 一个包含线性层、层归一化等操作的容器类。
    """
    def __init__(self, embed_dim, heads, dtype, device, operations):
        super().__init__()

        self.heads = heads # 注意力头的数量
        # 定义查询（Query）线性变换层
        self.q_proj = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        # 定义键（Key）线性变换层
        self.k_proj = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        # 定义值（Value）线性变换层
        self.v_proj = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)

        # 定义输出线性变换层
        self.out_proj = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x, mask=None, optimized_attention=None):
        """
        前向传播函数。

        参数:
        - x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, embed_dim)。
        - mask (Optional[torch.Tensor]): 可选的注意力掩码，形状为 (batch_size, sequence_length)。
        - optimized_attention (Optional[Any]): 可选的优化注意力机制。

        返回:
        - torch.Tensor: 输出张量，形状为 (batch_size, sequence_length, embed_dim)。
        """
        q = self.q_proj(x) # 计算查询（Query）投影
        k = self.k_proj(x) # 计算键（Key）投影
        v = self.v_proj(x) # 计算值（Value）投影
        
        # 使用优化后的注意力机制计算注意力输出
        out = optimized_attention(q, k, v, self.heads, mask)
        # 通过输出线性层进行投影
        return self.out_proj(out)


# 定义激活函数映射字典
ACTIVATIONS = {"quick_gelu": lambda a: a * torch.sigmoid(1.702 * a),
               "gelu": torch.nn.functional.gelu,
               "gelu_pytorch_tanh": lambda a: torch.nn.functional.gelu(a, approximate="tanh"),
}


class CLIPMLP(torch.nn.Module):
    """
    CLIP模型中的多层感知机（MLP）模块。

    参数:
    - embed_dim (int): 嵌入维度（embedding dimension）。
    - intermediate_size (int): 中间层的维度（intermediate layer dimension）。
    - activation (str): 激活函数类型，如 'quick_gelu'、'gelu'。
    - dtype (torch.dtype): 数据类型，如torch.float32。
    - device (torch.device): 计算设备，如'cuda'或'cpu'。
    - operations (Operations): 一个包含线性层、层归一化等操作的容器类。
    """
    def __init__(self, embed_dim, intermediate_size, activation, dtype, device, operations):
        super().__init__()
        # 第一个全连接层
        self.fc1 = operations.Linear(embed_dim, intermediate_size, bias=True, dtype=dtype, device=device)
        # 选择激活函数
        self.activation = ACTIVATIONS[activation]
        # 第二个全连接层
        self.fc2 = operations.Linear(intermediate_size, embed_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        """
        前向传播函数。

        参数:
        - x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, embed_dim)。

        返回:
        - torch.Tensor: 输出张量，形状为 (batch_size, sequence_length, embed_dim)。
        """
        # 通过第一个全连接层
        x = self.fc1(x)
        # 应用激活函数
        x = self.activation(x)
        # 通过第二个全连接层
        x = self.fc2(x)
        return x


class CLIPLayer(torch.nn.Module):
    """
    CLIP模型中的一层Transformer层。

    参数:
    - embed_dim (int): 嵌入维度（embedding dimension）。
    - heads (int): 注意力头的数量（number of attention heads）。
    - intermediate_size (int): 中间层的维度（intermediate layer dimension）。
    - intermediate_activation (str): 中间层的激活函数类型，如 'quick_gelu'、'gelu'。
    - dtype (torch.dtype): 数据类型，如torch.float32。
    - device (torch.device): 计算设备，如'cuda'或'cpu'。
    - operations (Operations): 一个包含线性层、层归一化等操作的容器类。
    """
    def __init__(self, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device, operations):
        super().__init__()
        # 第一个层归一化
        self.layer_norm1 = operations.LayerNorm(embed_dim, dtype=dtype, device=device)
        # 自注意力机制
        self.self_attn = CLIPAttention(embed_dim, heads, dtype, device, operations)
        # 第二个层归一化
        self.layer_norm2 = operations.LayerNorm(embed_dim, dtype=dtype, device=device)
        # MLP模块
        self.mlp = CLIPMLP(embed_dim, intermediate_size, intermediate_activation, dtype, device, operations)

    def forward(self, x, mask=None, optimized_attention=None):
        """
        前向传播函数。

        参数:
        - x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, embed_dim)。
        - mask (Optional[torch.Tensor]): 可选的注意力掩码，形状为 (batch_size, sequence_length)。
        - optimized_attention (Optional[Any]): 可选的优化注意力机制。

        返回:
        - torch.Tensor: 输出张量，形状为 (batch_size, sequence_length, embed_dim)。
        """
        # 自注意力机制
        x += self.self_attn(self.layer_norm1(x), mask, optimized_attention)
        # MLP模块
        x += self.mlp(self.layer_norm2(x))
        return x


class CLIPEncoder(torch.nn.Module):
    """
    CLIP模型的编码器部分，包含多层的Transformer层。

    参数:
    - num_layers (int): Transformer层的数量。
    - embed_dim (int): 嵌入维度（embedding dimension）。
    - heads (int): 注意力头的数量（number of attention heads）。
    - intermediate_size (int): 中间层的维度（intermediate layer dimension）。
    - intermediate_activation (str): 中间层的激活函数类型，如 'quick_gelu'、'gelu'。
    - dtype (torch.dtype): 数据类型，如torch.float32。
    - device (torch.device): 计算设备，如'cuda'或'cpu'。
    - operations (Operations): 一个包含线性层、层归一化等操作的容器类。
    """
    def __init__(self, num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device, operations):
        super().__init__()
        # 初始化Transformer层列表
        self.layers = torch.nn.ModuleList([CLIPLayer(embed_dim, heads, intermediate_size, intermediate_activation, dtype, device, operations) for i in range(num_layers)])

    def forward(self, x, mask=None, intermediate_output=None):
        """
        前向传播函数。

        参数:
        - x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, embed_dim)。
        - mask (Optional[torch.Tensor]): 可选的注意力掩码，形状为 (batch_size, sequence_length)。
        - intermediate_output (Optional[int]): 可选的中间输出层索引。如果提供，则返回指定层的输出。

        返回:
        - Tuple[torch.Tensor, Optional[torch.Tensor]]: 输出张量和可选的中间输出张量。
        """
        # 获取优化后的注意力机制
        optimized_attention = optimized_attention_for_device(x.device, mask=mask is not None, small_input=True)

        if intermediate_output is not None:
            if intermediate_output < 0:
                # 处理负数索引
                intermediate_output = len(self.layers) + intermediate_output

        intermediate = None
        for i, l in enumerate(self.layers):
            # 前向传播
            x = l(x, mask, optimized_attention)
            if i == intermediate_output:
                # 保存中间输出
                intermediate = x.clone()
        return x, intermediate


class CLIPEmbeddings(torch.nn.Module):
    """
    CLIP模型的嵌入层，将输入token和位置信息转换为嵌入向量。

    参数:
    - embed_dim (int): 嵌入维度（embedding dimension）。
    - vocab_size (int): 词汇表大小，默认为49408。
    - num_positions (int): 位置数量，默认为77。
    - dtype (torch.dtype): 数据类型，如torch.float32。
    - device (torch.device): 计算设备，如'cuda'或'cpu'。
    - operations (Operations): 一个包含嵌入层等操作的容器类。
    """
    def __init__(self, embed_dim, vocab_size=49408, num_positions=77, dtype=None, device=None, operations=None):
        super().__init__()
        # Token嵌入层
        self.token_embedding = operations.Embedding(vocab_size, embed_dim, dtype=dtype, device=device)
        # 位置嵌入层
        self.position_embedding = operations.Embedding(num_positions, embed_dim, dtype=dtype, device=device)

    def forward(self, input_tokens, dtype=torch.float32):
        """
        前向传播函数。

        参数:
        - input_tokens (torch.Tensor): 输入token张量，形状为 (batch_size, sequence_length)。
        - dtype (torch.dtype): 输出数据类型，默认为torch.float32。

        返回:
        - torch.Tensor: 嵌入向量张量，形状为 (batch_size, sequence_length, embed_dim)。
        """
        # 返回token和位置嵌入的和
        return self.token_embedding(input_tokens, out_dtype=dtype) + ops.cast_to(self.position_embedding.weight, dtype=dtype, device=input_tokens.device)


class CLIPTextModel_(torch.nn.Module):
    """
    CLIP模型的文本编码器部分。

    参数:
    - config_dict (dict): 配置字典，包含模型的各种参数，如层数、隐藏层大小等。
    - dtype (torch.dtype): 数据类型，如torch.float32。
    - device (torch.device): 计算设备，如'cuda'或'cpu'。
    - operations (Any): 一个包含层归一化、线性层等操作的容器类。
    """
    def __init__(self, config_dict, dtype, device, operations):
        num_layers = config_dict["num_hidden_layers"]
        embed_dim = config_dict["hidden_size"]
        heads = config_dict["num_attention_heads"]
        intermediate_size = config_dict["intermediate_size"]
        intermediate_activation = config_dict["hidden_act"]
        num_positions = config_dict["max_position_embeddings"]
        self.eos_token_id = config_dict["eos_token_id"]

        super().__init__()
        # 初始化嵌入层
        self.embeddings = CLIPEmbeddings(embed_dim, num_positions=num_positions, dtype=dtype, device=device, operations=operations)
        # 初始化编码器
        self.encoder = CLIPEncoder(num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device, operations)
        # 初始化最终的层归一化
        self.final_layer_norm = operations.LayerNorm(embed_dim, dtype=dtype, device=device)

    def forward(self, input_tokens, attention_mask=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=torch.float32):
        """
        前向传播函数。

        参数:
        - input_tokens (torch.Tensor): 输入token张量，形状为 (batch_size, sequence_length)。
        - attention_mask (Optional[torch.Tensor]): 可选的注意力掩码，形状为 (batch_size, sequence_length)。
        - intermediate_output (Optional[int]): 可选的中间输出层索引。
        - final_layer_norm_intermediate (bool): 是否对中间输出应用最终的层归一化。
        - dtype (torch.dtype): 数据类型，默认为torch.float32。

        返回:
        - Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]: 输出张量、中间输出张量和池化输出张量。
        """
        # 通过嵌入层获取嵌入向量
        x = self.embeddings(input_tokens, dtype=dtype)
        mask = None
        if attention_mask is not None:
            # 将注意力掩码转换为与嵌入向量相同的数据类型，并调整形状以匹配多头注意力的需求
            mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])).expand(attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1])
            # 使用掩码填充，将不需要注意力的位置设置为负无穷大
            mask = mask.masked_fill(mask.to(torch.bool), -torch.finfo(x.dtype).max)

        # 创建因果掩码，确保每个位置只能看到其左侧的上下文
        causal_mask = torch.full((x.shape[1], x.shape[1]), -torch.finfo(x.dtype).max, dtype=x.dtype, device=x.device).triu_(1)

        if mask is not None:
            # 将因果掩码添加到注意力掩码中
            mask += causal_mask
        else:
            # 如果没有提供注意力掩码，则仅使用因果掩码
            mask = causal_mask

        # 通过编码器进行前向传播
        x, i = self.encoder(x, mask=mask, intermediate_output=intermediate_output)
        # 应用最终的层归一化
        x = self.final_layer_norm(x)
        if i is not None and final_layer_norm_intermediate:
            # 如果需要，对中间输出应用层归一化
            i = self.final_layer_norm(i)

        # 计算池化输出，通常是EOS token的嵌入向量
        pooled_output = x[torch.arange(x.shape[0], device=x.device), (torch.round(input_tokens).to(dtype=torch.int, device=x.device) == self.eos_token_id).int().argmax(dim=-1),]
        # 返回输出张量、中间输出张量和池化输出张量
        return x, i, pooled_output


class CLIPTextModel(torch.nn.Module):
    """
    CLIP模型的文本模型封装。

    参数:
    - config_dict (dict): 配置字典，包含模型的各种参数。
    - dtype (torch.dtype): 数据类型，如torch.float32。
    - device (torch.device): 计算设备，如'cuda'或'cpu'。
    - operations (Any): 一个包含线性层等操作的容器类。
    """
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        # Transformer层的数量
        self.num_layers = config_dict["num_hidden_layers"]
        # 初始化文本编码器
        self.text_model = CLIPTextModel_(config_dict, dtype, device, operations)
        # 嵌入维度
        embed_dim = config_dict["hidden_size"]
        # 线性投影层
        self.text_projection = operations.Linear(embed_dim, embed_dim, bias=False, dtype=dtype, device=device)
        # 数据类型
        self.dtype = dtype

    def get_input_embeddings(self):
        """
        获取输入嵌入层。

        返回:
        - nn.Embedding: 输入嵌入层。
        """
        # 返回token嵌入层
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, embeddings):
        """
        设置输入嵌入层。

        参数:
        - embeddings (nn.Embedding): 新的token嵌入层。
        """
        # 设置token嵌入层
        self.text_model.embeddings.token_embedding = embeddings

    def forward(self, *args, **kwargs):
        """
        前向传播函数。

        参数:
        - *args: 位置参数，传递给文本编码器。
        - **kwargs: 关键字参数，传递给文本编码器。

        返回:
        - Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]: 输出张量、中间输出张量、投影输出张量和池化输出张量。
        """
        # 通过文本编码器进行前向传播
        x = self.text_model(*args, **kwargs)
        # 对池化输出进行线性投影
        out = self.text_projection(x[2])
        # 返回输出张量、中间输出张量、投影输出张量和池化输出张量
        return (x[0], x[1], out, x[2])


class CLIPVisionEmbeddings(torch.nn.Module):
    """
    CLIP模型的视觉嵌入层。

    参数:
    - embed_dim (int): 嵌入维度（embedding dimension）。
    - num_channels (int): 输入图像的通道数，默认为3。
    - patch_size (int): 图像块大小，默认为14。
    - image_size (int): 图像大小，默认为224。
    - model_type (str): 模型类型，默认为空字符串。
    - dtype (torch.dtype): 数据类型，如torch.float32。
    - device (torch.device): 计算设备，如'cuda'或'cpu'。
    - operations (Any): 一个包含卷积层、嵌入层等操作的容器类。
    """
    def __init__(self, embed_dim, num_channels=3, patch_size=14, image_size=224, model_type="", dtype=None, device=None, operations=None):
        super().__init__()

        # 计算图像块数量
        num_patches = (image_size // patch_size) ** 2
        if model_type == "siglip_vision_model":
            # 如果是siglip_vision_model，则没有类别嵌入
            self.class_embedding = None
            # 使用偏置
            patch_bias = True
        else:
            # 增加一个类别嵌入
            num_patches = num_patches + 1
            # 类别嵌入参数
            self.class_embedding = torch.nn.Parameter(torch.empty(embed_dim, dtype=dtype, device=device))
            # 不使用偏置
            patch_bias = False

        # 初始化卷积层，用于将图像分割成小块并嵌入
        self.patch_embedding = operations.Conv2d(
            in_channels=num_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=patch_bias,
            dtype=dtype,
            device=device
        )

        # 初始化位置嵌入层
        self.position_embedding = operations.Embedding(num_patches, embed_dim, dtype=dtype, device=device)

    def forward(self, pixel_values):
        """
        前向传播函数。

        参数:
        - pixel_values (torch.Tensor): 输入图像像素值，形状为 (batch_size, channels, height, width)。

        返回:
        - torch.Tensor: 视觉嵌入向量，形状为 (batch_size, num_patches, embed_dim)。
        """
        # 通过卷积层嵌入图像并调整形状
        embeds = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        if self.class_embedding is not None:
            # 添加类别嵌入
            embeds = torch.cat([ops.cast_to_input(self.class_embedding, embeds).expand(pixel_values.shape[0], 1, -1), embeds], dim=1)
        # 添加位置嵌入
        return embeds + ops.cast_to_input(self.position_embedding.weight, embeds)


class CLIPVision(torch.nn.Module):
    """
    CLIP模型的视觉模型部分。

    参数:
    - config_dict (dict): 配置字典，包含模型的各种参数。
    - dtype (torch.dtype): 数据类型，如torch.float32。
    - device (torch.device): 计算设备，如'cuda'或'cpu'。
    - operations (Any): 一个包含层归一化、线性层等操作的容器类。
    """
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        num_layers = config_dict["num_hidden_layers"]
        embed_dim = config_dict["hidden_size"]
        heads = config_dict["num_attention_heads"]
        intermediate_size = config_dict["intermediate_size"]
        intermediate_activation = config_dict["hidden_act"]
        model_type = config_dict["model_type"]

        self.embeddings = CLIPVisionEmbeddings(embed_dim, config_dict["num_channels"], config_dict["patch_size"], config_dict["image_size"], model_type=model_type, dtype=dtype, device=device, operations=operations)
        if model_type == "siglip_vision_model":
            # 如果是siglip_vision_model，则不使用预层归一化
            self.pre_layrnorm = lambda a: a
            # 使用输出层归一化
            self.output_layernorm = True
        else:
            # 初始化预层归一化
            self.pre_layrnorm = operations.LayerNorm(embed_dim)
            # 不使用输出层归一化
            self.output_layernorm = False
        self.encoder = CLIPEncoder(num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device, operations)
        # 初始化后层归一化
        self.post_layernorm = operations.LayerNorm(embed_dim)

    def forward(self, pixel_values, attention_mask=None, intermediate_output=None):
        """
        前向传播函数。

        参数:
        - pixel_values (torch.Tensor): 输入图像像素值，形状为 (batch_size, channels, height, width)。
        - attention_mask (Optional[torch.Tensor]): 可选的注意力掩码。
        - intermediate_output (Optional[int]): 可选的中间输出层索引。

        返回:
        - Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]: 输出张量、中间输出张量和池化输出张量。
        """
        # 通过视觉嵌入层获取嵌入向量
        x = self.embeddings(pixel_values)
        # 应用预层归一化
        x = self.pre_layrnorm(x)
        # 通过编码器进行前向传播
        x, i = self.encoder(x, mask=None, intermediate_output=intermediate_output)
        if self.output_layernorm:
            # 应用后层归一化
            x = self.post_layernorm(x)
            # 池化输出为整个输出张量
            pooled_output = x
        else:
            # 池化输出为第一个token的嵌入向量
            pooled_output = self.post_layernorm(x[:, 0, :])
        # 返回输出张量、中间输出张量和池化输出张量
        return x, i, pooled_output


class CLIPVisionModelProjection(torch.nn.Module):
    """
    CLIP模型的视觉投影模块。

    参数:
    - config_dict (dict): 配置字典，包含模型的各种参数。
    - dtype (torch.dtype): 数据类型，如torch.float32。
    - device (torch.device): 计算设备，如'cuda'或'cpu'。
    - operations (Any): 一个包含线性层等操作的容器类。
    """
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        # 初始化视觉模型
        self.vision_model = CLIPVision(config_dict, dtype, device, operations)
        if "projection_dim" in config_dict:
            # 线性投影层
            self.visual_projection = operations.Linear(config_dict["hidden_size"], config_dict["projection_dim"], bias=False)
        else:
            # 如果没有指定projection_dim，则不进行投影
            self.visual_projection = lambda a: a

    def forward(self, *args, **kwargs):
        """
        前向传播函数。

        参数:
        - *args: 位置参数，传递给视觉模型。
        - **kwargs: 关键字参数，传递给视觉模型。

        返回:
        - Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]: 输出张量、中间输出张量和投影输出张量。
        """
        # 通过视觉模型进行前向传播
        x = self.vision_model(*args, **kwargs)
        # 对池化输出进行线性投影
        out = self.visual_projection(x[2])

        # 返回输出张量、中间输出张量和投影输出张量
        return (x[0], x[1], out)
