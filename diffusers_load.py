import os
import sd


def first_file(path, filenames):
    """
    在指定路径下查找并返回第一个存在的文件名对应的完整路径。
    按顺序遍历给定的文件名列表，找到第一个存在的文件并返回其路径。
    如果没有找到任何存在的文件，则返回 None。

    参数:
        path (str): 要搜索的目录路径。
        filenames (list of str): 要查找的文件名列表。

    返回:
        str 或 None: 第一个存在的文件的完整路径，或如果没有找到则返回 None。
    """
    for f in filenames:
        # 构建文件的完整路径
        p = os.path.join(path, f)
        # 检查文件是否存在
        if os.path.exists(p):
            # 返回存在的文件的完整路径
            return p
    return None


def load_diffusers(model_path, output_vae=True, output_clip=True, embedding_directory=None):
    """
    加载扩散模型的组件，包括 UNet、VAE 和 CLIP（文本编码器）。
    根据提供的模型路径，加载必要的模型文件，并初始化相应的组件。

    参数:
        model_path (str): 包含模型组件的根目录路径。
        output_vae (bool, 可选): 是否加载 VAE，默认为 True。
        output_clip (bool, 可选): 是否加载 CLIP（文本编码器），默认为 True。
        embedding_directory (str, 可选): 嵌入目录的路径，默认为 None。

    返回:
        tuple: 包含 UNet、CLIP（如果加载）和 VAE（如果加载）的元组。
               例如: (unet, clip, vae)
    """
    # 定义扩散模型中 UNet 组件的可能文件名列表
    diffusion_model_names = ["diffusion_pytorch_model.fp16.safetensors", "diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.fp16.bin", "diffusion_pytorch_model.bin"]
    
    # 构建 UNet 组件的搜索路径，并查找第一个存在的文件
    unet_path = first_file(os.path.join(model_path, "unet"), diffusion_model_names)
    
    # 构建 VAE 组件的搜索路径，并查找第一个存在的文件
    vae_path = first_file(os.path.join(model_path, "vae"), diffusion_model_names)

    # 定义 CLIP（文本编码器）组件的可能文件名列表
    text_encoder_model_names = ["model.fp16.safetensors", "model.safetensors", "pytorch_model.fp16.bin", "pytorch_model.bin"]
    
    # 构建第一个文本编码器（text_encoder）的搜索路径，并查找第一个存在的文件
    text_encoder1_path = first_file(os.path.join(model_path, "text_encoder"), text_encoder_model_names)
    
    # 构建第二个文本编码器（text_encoder_2）的搜索路径，并查找第一个存在的文件
    text_encoder2_path = first_file(os.path.join(model_path, "text_encoder_2"), text_encoder_model_names)

    # 初始化文本编码器路径列表，包含第一个文本编码器路径
    text_encoder_paths = [text_encoder1_path]

    # 如果第二个文本编码器路径存在，则将其添加到列表中
    if text_encoder2_path is not None:
        text_encoder_paths.append(text_encoder2_path)

    # 加载 UNet 组件
    unet = sd.load_diffusion_model(unet_path)

    # 初始化 CLIP（文本编码器）为 None
    clip = None
    # 如果需要输出 CLIP，则加载文本编码器
    if output_clip:
        clip = sd.load_clip(text_encoder_paths, embedding_directory=embedding_directory)

    # 初始化 VAE 为 None
    vae = None

    # 如果需要输出 VAE，则加载 VAE
    if output_vae:
        # 加载 VAE 的 Torch 文件
        sd = utils.load_torch_file(vae_path)
        vae = sd.VAE(sd=sd)

    # 返回包含 UNet、CLIP 和 VAE 的元组
    return (unet, clip, vae)
