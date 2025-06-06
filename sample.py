import torch
import model_management
import samplers
import utils
import numpy as np
import logging


def prepare_noise(latent_image, seed, noise_inds=None):
    """
    根据给定的潜在图像和种子生成随机噪声。
    可选参数 `noise_inds` 用于跳过并丢弃指定数量的噪声生成，以实现特定的噪声序列。

    参数:
        latent_image (torch.Tensor): 潜在图像张量，用于确定噪声的形状和数据类型。
        seed (int): 随机数生成的种子，确保可重复性。
        noise_inds (list or np.ndarray, 可选): 要生成的噪声索引列表。如果提供，将根据这些索引生成噪声。

    返回:
        torch.Tensor: 生成的噪声张量。
    """
    # 设置随机数生成器的种子，以确保每次调用生成相同的噪声
    generator = torch.manual_seed(seed)

    # 如果没有提供 `noise_inds`，则生成一个与 `latent_image` 形状相同的标准正态分布噪声张量
    if noise_inds is None:
        return torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu")

    # 处理 `noise_inds`，获取唯一的索引及其反向索引映射
    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []

    # 遍历所有唯一的索引，生成对应数量的噪声
    for i in range(unique_inds[-1]+1):
        # 生成一个噪声张量，形状为 [1, latent_image.shape[1:], ...]
        noise = torch.randn([1] + list(latent_image.size())[1:], dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu")
        # 如果当前索引在 `unique_inds` 中，则将其添加到 `noises` 列表中
        if i in unique_inds:
            noises.append(noise)

    # 根据 `inverse` 映射重新排列噪声列表，以匹配原始的 `noise_inds` 顺序
    noises = [noises[i] for i in inverse]
    # 将所有噪声张量在第0维（批次维度）上进行拼接
    noises = torch.cat(noises, axis=0)
    return noises


def fix_empty_latent_channels(model, latent_image):
    """
    修正空的潜在图像通道数，以确保其与模型期望的通道数一致。
    如果潜在图像为空且通道数不匹配，则重复潜在图像以匹配期望的通道数。
    如果模型的潜在维度为3且潜在图像的维度为4，则在第2维添加一个维度。

    参数:
        model: 使用的模型对象，必须具有 `get_model_object` 方法以获取 `latent_format`。
        latent_image (torch.Tensor): 潜在图像张量。

    返回:
        torch.Tensor: 修正后的潜在图像张量。
    """
    # 从模型中获取 `latent_format` 对象，该对象包含有关潜在图像格式的信息
    latent_format = model.get_model_object("latent_format") #Resize the empty latent image so it has the right number of channels
    
    # 检查潜在图像的通道数是否与模型期望的通道数不同，且潜在图像是否为空
    if latent_format.latent_channels != latent_image.shape[1] and torch.count_nonzero(latent_image) == 0:
        # 使用 `repeat_to_batch_size` 函数重复潜在图像以匹配期望的通道数
        latent_image = utils.repeat_to_batch_size(latent_image, latent_format.latent_channels, dim=1)

    if latent_format.latent_dimensions == 3 and latent_image.ndim == 4:
        # 如果模型的潜在维度为3且潜在图像的维度为4，则在第2维添加一个维度
        latent_image = latent_image.unsqueeze(2)
    return latent_image


def prepare_sampling(model, noise_shape, positive, negative, noise_mask):
    """
    准备采样过程（已弃用）。
    此函数用于准备采样参数，但由于其已不再使用，建议移除。

    参数:
        model: 使用的模型对象。
        noise_shape (tuple): 噪声的形状。
        positive: 正向条件或输入。
        negative: 负向条件或输入。
        noise_mask (torch.Tensor, 可选): 噪声掩码。

    返回:
        tuple: 包含模型、正向输入、负向输入、噪声掩码和空列表的元组。
    """
    logging.warning("Warning: comfy.sample.prepare_sampling isn't used anymore and can be removed")
    return model, positive, negative, noise_mask, []


def cleanup_additional_models(models):
    """
    清理额外的模型（已弃用）。
    此函数用于清理额外的模型，但由于其已不再使用，建议移除。

    参数:
        models (list): 模型列表。
    """
    logging.warning("Warning: comfy.sample.cleanup_additional_models isn't used anymore and can be removed")


def sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
    """
    使用指定的采样器对噪声进行采样，生成潜在图像。

    参数:
        model: 使用的模型对象。
        noise (torch.Tensor): 输入噪声张量。
        steps (int): 采样步骤数。
        cfg (float): 分类器自由引导（Classifier-Free Guidance）因子。
        sampler_name (str): 采样器的名称。
        scheduler: 调度器对象，用于控制采样过程。
        positive: 正向条件或输入。
        negative: 负向条件或输入。
        latent_image (torch.Tensor): 初始潜在图像张量。
        denoise (float, 可选): 去噪因子，默认为1.0。
        disable_noise (bool, 可选): 是否禁用噪声，默认为False。
        start_step (int, 可选): 开始的采样步骤，默认为None。
        last_step (int, 可选): 结束的采样步骤，默认为None。
        force_full_denoise (bool, 可选): 是否强制完全去噪，默认为False。
        noise_mask (torch.Tensor, 可选): 噪声掩码，默认为None。
        sigmas (list or tuple, 可选): 标准差列表，默认为None。
        callback (function, 可选): 回调函数，默认为None。
        disable_pbar (bool, 可选): 是否禁用进度条，默认为False。
        seed (int, 可选): 随机数种子，默认为None。

    返回:
        torch.Tensor: 采样后的潜在图像张量。
    """
    # 初始化采样器，使用指定的参数
    sampler = samplers.KSampler(model, steps=steps, device=model.load_device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)

    # 执行采样过程，传入所有必要的参数
    samples = sampler.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)
    # 将采样结果移动到中间设备（假设这是模型管理的一部分）
    samples = samples.to(model_management.intermediate_device())
    return samples


def sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image, noise_mask=None, callback=None, disable_pbar=False, seed=None):
    """
    使用自定义采样器对噪声进行采样，生成潜在图像。

    参数:
        model: 使用的模型对象。
        noise (torch.Tensor): 输入噪声张量。
        cfg (float): 分类器自由引导（Classifier-Free Guidance）因子。
        sampler (str): 采样器的名称。
        sigmas (list or tuple): 标准差列表。
        positive: 正向条件或输入。
        negative: 负向条件或输入。
        latent_image (torch.Tensor): 初始潜在图像张量。
        noise_mask (torch.Tensor, 可选): 噪声掩码，默认为None。
        callback (function, 可选): 回调函数，默认为None。
        disable_pbar (bool, 可选): 是否禁用进度条，默认为False。
        seed (int, 可选): 随机数种子，默认为None。

    返回:
        torch.Tensor: 采样后的潜在图像张量。
    """
    # 使用自定义采样器执行采样过程，传入所有必要的参数
    samples = samplers.sample(model, noise, positive, negative, cfg, model.load_device, sampler, sigmas, model_options=model.model_options, latent_image=latent_image, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    # 将采样结果移动到中间设备
    samples = samples.to(model_management.intermediate_device())
    return samples
