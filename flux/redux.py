import torch
import ops

ops = ops.manual_cast


class ReduxImageEncoder(torch.nn.Module):
    """
    ReduxImageEncoder 类，用于对图像特征进行降维和投影。
    该类通过线性层将输入的 Redux 特征（redux_dim）映射到文本输入特征空间（txt_in_features），
    然后再将其投影回文本输入特征空间，以实现特征对齐和降维。
    """
    def __init__(
        self,
        redux_dim: int = 1152,
        txt_in_features: int = 4096,
        device=None,
        dtype=None,
    ) -> None:
        """
        初始化 ReduxImageEncoder 模块。

        参数:
            redux_dim (int, 可选): Redux 特征的维度，默认为 1152。
            txt_in_features (int, 可选): 文本输入特征的维度，默认为 4096。
            device (torch.device, 可选): 张量所在的设备，默认为 None。
            dtype (torch.dtype, 可选): 张量的数据类型，默认为 None。
        """
        # 调用父类初始化方法
        super().__init__()

        # 设置 Redux 特征的维度
        self.redux_dim = redux_dim
        self.device = device
        # 设置数据类型
        self.dtype = dtype

        # 定义上投影线性层，将 Redux 特征映射到文本输入特征空间的 3 倍维度
        self.redux_up = ops.Linear(redux_dim, txt_in_features * 3, dtype=dtype)
        # 定义下投影线性层，将上投影后的特征映射回文本输入特征空间
        self.redux_down = ops.Linear(txt_in_features * 3, txt_in_features, dtype=dtype)

    def forward(self, sigclip_embeds) -> torch.Tensor:
        """
        前向传播函数，对输入的 Redux 特征进行降维和投影。

        参数:
            sigclip_embeds (torch.Tensor): 输入的 Redux 特征张量，形状为 (batch_size, redux_dim)。

        返回:
            torch.Tensor: 投影后的特征张量，形状为 (batch_size, txt_in_features)。
        """
        # 将输入张量通过上投影线性层进行投影
        # 应用 SiLU 激活函数
        # 将激活后的张量通过下投影线性层进行投影
        projected_x = self.redux_down(torch.nn.functional.silu(self.redux_up(sigclip_embeds)))
        return projected_x
