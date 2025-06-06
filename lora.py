from __future__ import annotations
import utils
import model_management
import model_base
import weight_adapter as weight_adapter
import logging
import torch


LORA_CLIP_MAP = {
    "mlp.fc1": "mlp_fc1",
    "mlp.fc2": "mlp_fc2",
    "self_attn.k_proj": "self_attn_k_proj",
    "self_attn.q_proj": "self_attn_q_proj",
    "self_attn.v_proj": "self_attn_v_proj",
    "self_attn.out_proj": "self_attn_out_proj",
}


def load_lora(lora, to_load, log_missing=True):
    """
    从 Lora 状态字典中加载指定的 Lora 参数，并构建一个补丁字典。
    该函数会遍历 `to_load` 中指定的键，查找对应的 Lora 参数（如 alpha、dora_scale 等），
    并将找到的参数组织到一个字典中，以便后续应用补丁。

    参数:
        lora (dict): 包含 Lora 参数的字典。
        to_load (dict): 需要加载的 Lora 参数的映射字典，键是需要加载的参数名，值是目标参数名。
        log_missing (bool, 可选): 是否记录未加载的 Lora 键的警告，默认为 True。

    返回:
        dict: 包含补丁信息的字典，格式为 {目标参数名: (操作类型, 参数元组)}。
    """
    patch_dict = {}
    loaded_keys = set()
    for x in to_load:
        # 构建 alpha 参数的名称，并尝试从 Lora 字典中获取其值
        alpha_name = "{}.alpha".format(x)
        alpha = None
        if alpha_name in lora.keys():
            alpha = lora[alpha_name].item()
            loaded_keys.add(alpha_name)

        # 构建 dora_scale 参数的名称，并尝试从 Lora 字典中获取其值
        dora_scale_name = "{}.dora_scale".format(x)
        dora_scale = None
        if dora_scale_name in lora.keys():
            dora_scale = lora[dora_scale_name]
            loaded_keys.add(dora_scale_name)

        # 遍历所有支持的权重适配器类，尝试加载相应的适配器
        for adapter_cls in weight_adapter.adapters:
            adapter = adapter_cls.load(x, lora, alpha, dora_scale, loaded_keys)
            if adapter is not None:
                # 将加载的适配器添加到补丁字典中，键为目标参数名
                patch_dict[to_load[x]] = adapter
                # 更新已加载的键集合
                loaded_keys.update(adapter.loaded_keys)
                # 继续下一个适配器（如果有的话）
                continue
        
        # 构建权重归一化参数名称，并尝试从 Lora 字典中获取其值
        w_norm_name = "{}.w_norm".format(x)
        b_norm_name = "{}.b_norm".format(x)
        w_norm = lora.get(w_norm_name, None)
        b_norm = lora.get(b_norm_name, None)

        if w_norm is not None:
            loaded_keys.add(w_norm_name)
            # 将权重归一化参数添加到补丁字典中，类型为 "diff"
            patch_dict[to_load[x]] = ("diff", (w_norm,))
            if b_norm is not None:
                # 构建偏置归一化参数名称，并尝试从 Lora 字典中获取其值
                loaded_keys.add(b_norm_name)
                # 将偏置归一化参数添加到补丁字典中，键为目标参数的偏置名
                patch_dict["{}.bias".format(to_load[x][:-len(".weight")])] = ("diff", (b_norm,))

        # 构建差分权重参数名称，并尝试从 Lora 字典中获取其值
        diff_name = "{}.diff".format(x)
        diff_weight = lora.get(diff_name, None)
        if diff_weight is not None:
            patch_dict[to_load[x]] = ("diff", (diff_weight,))
            loaded_keys.add(diff_name)

        # 构建差分偏置参数名称，并尝试从 Lora 字典中获取其值
        diff_bias_name = "{}.diff_b".format(x)
        diff_bias = lora.get(diff_bias_name, None)
        if diff_bias is not None:
            patch_dict["{}.bias".format(to_load[x][:-len(".weight")])] = ("diff", (diff_bias,))
            loaded_keys.add(diff_bias_name)

        # 构建设置权重参数名称，并尝试从 Lora 字典中获取其值
        set_weight_name = "{}.set_weight".format(x)
        set_weight = lora.get(set_weight_name, None)
        if set_weight is not None:
            patch_dict[to_load[x]] = ("set", (set_weight,))
            loaded_keys.add(set_weight_name)
    
    # 如果启用了日志记录，则记录所有未加载的 Lora 键
    if log_missing:
        for x in lora.keys():
            if x not in loaded_keys:
                logging.warning("lora key not loaded: {}".format(x))

    return patch_dict


def model_lora_keys_clip(model, key_map={}):
    """
    根据模型的状态字典，构建一个 Lora 键映射字典，将模型中的键名映射到 Lora 键名。
    该函数主要处理 CLIP 相关的层，并生成相应的 Lora 键名。

    参数:
        model: 要处理的模型对象，必须具有 `state_dict()` 方法。
        key_map (dict, 可选): 现有的键映射字典，默认为空字典。

    返回:
        dict: 包含键映射的字典，格式为 {Lora 键名: 模型键名}。
    """
    sdk = model.state_dict().keys()
    # 遍历模型的状态字典中的所有键
    for k in sdk:
        if k.endswith(".weight"):
            # 将模型中的键名映射到通用的 Lora 格式键名
            key_map["text_encoders.{}".format(k[:-len(".weight")])] = k #generic lora format without any weird key names

    # 定义文本模型 Lora 键的格式
    text_model_lora_key = "lora_te_text_model_encoder_layers_{}_{}"
    clip_l_present = False
    clip_g_present = False

    # 遍历 CLIP 层的索引
    for b in range(32): #TODO: clean up
        for c in LORA_CLIP_MAP:
            # 构建 CLIP 层的键名，并检查其是否存在于模型状态字典中
            k = "clip_h.transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                # 生成对应的 Lora 键名，并添加到键映射字典中
                lora_key = text_model_lora_key.format(b, LORA_CLIP_MAP[c])
                key_map[lora_key] = k
                lora_key = "lora_te1_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c])
                key_map[lora_key] = k
                lora_key = "text_encoder.text_model.encoder.layers.{}.{}".format(b, c) #diffusers lora
                key_map[lora_key] = k

            # 同上，处理另一个 CLIP 层
            k = "clip_l.transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                lora_key = text_model_lora_key.format(b, LORA_CLIP_MAP[c])
                key_map[lora_key] = k
                lora_key = "lora_te1_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c]) #SDXL base
                key_map[lora_key] = k
                clip_l_present = True
                lora_key = "text_encoder.text_model.encoder.layers.{}.{}".format(b, c) #diffusers lora
                key_map[lora_key] = k

            # 同上，处理第三个 CLIP 层
            k = "clip_g.transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                clip_g_present = True
                if clip_l_present:
                    lora_key = "lora_te2_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c]) #SDXL base
                    key_map[lora_key] = k
                    lora_key = "text_encoder_2.text_model.encoder.layers.{}.{}".format(b, c) #diffusers lora
                    key_map[lora_key] = k
                else:
                    lora_key = "lora_te_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c]) #TODO: test if this is correct for SDXL-Refiner
                    key_map[lora_key] = k
                    lora_key = "text_encoder.text_model.encoder.layers.{}.{}".format(b, c) #diffusers lora
                    key_map[lora_key] = k
                    lora_key = "lora_prior_te_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c]) #cascade lora: TODO put lora key prefix in the model config
                    key_map[lora_key] = k

    # 处理其他特定的键名
    for k in sdk:
        if k.endswith(".weight"):
            if k.startswith("t5xxl.transformer."):#OneTrainer SD3 and Flux lora
                l_key = k[len("t5xxl.transformer."):-len(".weight")]
                t5_index = 1
                if clip_g_present:
                    t5_index += 1
                if clip_l_present:
                    t5_index += 1
                    if t5_index == 2:
                        key_map["lora_te{}_{}".format(t5_index, l_key.replace(".", "_"))] = k #OneTrainer Flux
                        t5_index += 1

                key_map["lora_te{}_{}".format(t5_index, l_key.replace(".", "_"))] = k
            elif k.startswith("hydit_clip.transformer.bert."): #HunyuanDiT Lora
                l_key = k[len("hydit_clip.transformer.bert."):-len(".weight")]
                lora_key = "lora_te1_{}".format(l_key.replace(".", "_"))
                key_map[lora_key] = k

    # 处理文本投影权重键名
    k = "clip_g.transformer.text_projection.weight"
    if k in sdk:
        key_map["lora_prior_te_text_projection"] = k #cascade lora?
        # key_map["text_encoder.text_projection"] = k #TODO: check if other lora have the text_projection too
        key_map["lora_te2_text_projection"] = k #OneTrainer SD3 lora

    # 处理另一个文本投影权重键名
    k = "clip_l.transformer.text_projection.weight"
    if k in sdk:
        key_map["lora_te1_text_projection"] = k #OneTrainer SD3 lora, not necessary but omits warning

    return key_map


def model_lora_keys_unet(model, key_map={}):
    """
    根据模型的状态字典，构建一个 Lora 键映射字典，将 UNet 相关的键名映射到 Lora 键名。
    该函数处理不同类型的模型，并生成相应的 Lora 键名。

    参数:
        model: 要处理的模型对象，必须具有 `state_dict()` 方法。
        key_map (dict, 可选): 现有的键映射字典，默认为空字典。

    返回:
        dict: 包含键映射的字典，格式为 {Lora 键名: 模型键名}。
    """
    sd = model.state_dict()
    sdk = sd.keys()

    for k in sdk:
        if k.startswith("diffusion_model."):
            if k.endswith(".weight"):
                # 移除前缀 "diffusion_model." 和后缀 ".weight"，并用 "_" 替换 "."
                key_lora = k[len("diffusion_model."):-len(".weight")].replace(".", "_")
                key_map["lora_unet_{}".format(key_lora)] = k
                # 通用的 Lora 格式，不包含任何奇怪的键名
                key_map["{}".format(k[:-len(".weight")])] = k #generic lora format without any weird key names
            else:
                # 通用的 Lora 格式，适用于非 ".weight" 键名
                key_map["{}".format(k)] = k #generic lora format for not .weight without any weird key names

    # 将 UNet 配置转换为 Diffusers 键名
    diffusers_keys = utils.unet_to_diffusers(model.model_config.unet_config)
    for k in diffusers_keys:
        if k.endswith(".weight"):
            # 构建 UNet 键名
            unet_key = "diffusion_model.{}".format(diffusers_keys[k])
            key_lora = k[:-len(".weight")].replace(".", "_")
            key_map["lora_unet_{}".format(key_lora)] = unet_key
            key_map["lycoris_{}".format(key_lora)] = unet_key #simpletuner lycoris format

            # 定义 Diffusers Lora 前缀列表
            diffusers_lora_prefix = ["", "unet."]
            for p in diffusers_lora_prefix:
                diffusers_lora_key = "{}{}".format(p, k[:-len(".weight")].replace(".to_", ".processor.to_"))
                if diffusers_lora_key.endswith(".to_out.0"):
                    diffusers_lora_key = diffusers_lora_key[:-2]
                key_map[diffusers_lora_key] = unet_key

    # 如果模型是 StableCascade_C 类型，则处理特定键名
    if isinstance(model, model_base.StableCascade_C):
        for k in sdk:
            if k.startswith("diffusion_model."):
                if k.endswith(".weight"):
                    key_lora = k[len("diffusion_model."):-len(".weight")].replace(".", "_")
                    key_map["lora_prior_unet_{}".format(key_lora)] = k

    # 如果模型是 SD3 类型，则处理 Diffusers Lora SD3 格式
    if isinstance(model, model_base.SD3): #Diffusers lora SD3
        diffusers_keys = utils.mmdit_to_diffusers(model.model_config.unet_config, output_prefix="diffusion_model.")
        for k in diffusers_keys:
            if k.endswith(".weight"):
                to = diffusers_keys[k]
                key_lora = "transformer.{}".format(k[:-len(".weight")]) #regular diffusers sd3 lora format
                key_map[key_lora] = to

                key_lora = "base_model.model.{}".format(k[:-len(".weight")]) #format for flash-sd3 lora and others?
                key_map[key_lora] = to

                key_lora = "lora_transformer_{}".format(k[:-len(".weight")].replace(".", "_")) #OneTrainer lora
                key_map[key_lora] = to

                key_lora = "lycoris_{}".format(k[:-len(".weight")].replace(".", "_")) #simpletuner lycoris format
                key_map[key_lora] = to

    # 如果模型是 AuraFlow 类型，则处理 Diffusers Lora AuraFlow 格式
    if isinstance(model, model_base.AuraFlow): #Diffusers lora AuraFlow
        diffusers_keys = utils.auraflow_to_diffusers(model.model_config.unet_config, output_prefix="diffusion_model.")
        for k in diffusers_keys:
            if k.endswith(".weight"):
                to = diffusers_keys[k]
                key_lora = "transformer.{}".format(k[:-len(".weight")]) #simpletrainer and probably regular diffusers lora format
                key_map[key_lora] = to

    # 如果模型是 PixArt 类型，则处理 Diffusers Lora PixArt 格式
    if isinstance(model, model_base.PixArt):
        diffusers_keys = utils.pixart_to_diffusers(model.model_config.unet_config, output_prefix="diffusion_model.")
        for k in diffusers_keys:
            if k.endswith(".weight"):
                to = diffusers_keys[k]
                key_lora = "transformer.{}".format(k[:-len(".weight")]) #default format
                key_map[key_lora] = to

                key_lora = "base_model.model.{}".format(k[:-len(".weight")]) #diffusers training script
                key_map[key_lora] = to

                key_lora = "unet.base_model.model.{}".format(k[:-len(".weight")]) #old reference peft script
                key_map[key_lora] = to

    # 如果模型是 HunyuanDiT 类型，则处理特定键名
    if isinstance(model, model_base.HunyuanDiT):
        for k in sdk:
            if k.startswith("diffusion_model.") and k.endswith(".weight"):
                key_lora = k[len("diffusion_model."):-len(".weight")]
                key_map["base_model.model.{}".format(key_lora)] = k #official hunyuan lora format

    # 如果模型是 Flux 类型，则处理 Diffusers Lora Flux 格式
    if isinstance(model, model_base.Flux): #Diffusers lora Flux
        diffusers_keys = utils.flux_to_diffusers(model.model_config.unet_config, output_prefix="diffusion_model.")
        for k in diffusers_keys:
            if k.endswith(".weight"):
                to = diffusers_keys[k]
                key_map["transformer.{}".format(k[:-len(".weight")])] = to #simpletrainer and probably regular diffusers flux lora format
                key_map["lycoris_{}".format(k[:-len(".weight")].replace(".", "_"))] = to #simpletrainer lycoris
                key_map["lora_transformer_{}".format(k[:-len(".weight")].replace(".", "_"))] = to #onetrainer

    # 如果模型是 GenmoMochi 类型，则处理特定键名
    if isinstance(model, model_base.GenmoMochi):
        for k in sdk:
            if k.startswith("diffusion_model.") and k.endswith(".weight"): #Official Mochi lora format
                key_lora = k[len("diffusion_model."):-len(".weight")]
                key_map["{}".format(key_lora)] = k

    # 如果模型是 HunyuanVideo 类型，则处理特定键名
    if isinstance(model, model_base.HunyuanVideo):
        for k in sdk:
            if k.startswith("diffusion_model.") and k.endswith(".weight"):
                # diffusion-pipe lora format
                key_lora = k
                key_lora = key_lora.replace("_mod.lin.", "_mod.linear.").replace("_attn.qkv.", "_attn_qkv.").replace("_attn.proj.", "_attn_proj.")
                key_lora = key_lora.replace("mlp.0.", "mlp.fc1.").replace("mlp.2.", "mlp.fc2.")
                key_lora = key_lora.replace(".modulation.lin.", ".modulation.linear.")
                key_lora = key_lora[len("diffusion_model."):-len(".weight")]
                key_map["transformer.{}".format(key_lora)] = k
                key_map["diffusion_model.{}".format(key_lora)] = k  # Old loras

    # 如果模型是 HiDream 类型，则处理特定键名
    if isinstance(model, model_base.HiDream):
        for k in sdk:
            if k.startswith("diffusion_model."):
                if k.endswith(".weight"):
                    key_lora = k[len("diffusion_model."):-len(".weight")]
                    key_map["lycoris_{}".format(key_lora.replace(".", "_"))] = k #SimpleTuner lycoris format
                    key_map["transformer.{}".format(key_lora)] = k #SimpleTuner regular format

    # 如果模型是 ACEStep 类型，则处理特定键名
    if isinstance(model, model_base.ACEStep):
        for k in sdk:
            if k.startswith("diffusion_model.") and k.endswith(".weight"): #Official ACE step lora format
                key_lora = k[len("diffusion_model."):-len(".weight")]
                key_map["{}".format(key_lora)] = k

    return key_map


def pad_tensor_to_shape(tensor: torch.Tensor, new_shape: list[int]) -> torch.Tensor:
    """
    将张量填充到新的形状，填充值为零。

    参数:
        tensor (torch.Tensor): 要填充的原始张量。
        new_shape (List[int]): 填充后的目标形状。

    返回:
        torch.Tensor: 填充后的新张量。

    注意:
        如果新的形状在任何一个维度上小于原始张量，则会引发 ValueError。
    """
    if any([new_shape[i] < tensor.shape[i] for i in range(len(new_shape))]):
        raise ValueError("The new shape must be larger than the original tensor in all dimensions")

    if len(new_shape) != len(tensor.shape):
        raise ValueError("The new shape must have the same number of dimensions as the original tensor")

    # 创建一个填充为零的新张量
    padded_tensor = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)

    # 创建原始张量和填充后张量的切片元组
    orig_slices = tuple(slice(0, dim) for dim in tensor.shape)
    new_slices = tuple(slice(0, dim) for dim in tensor.shape)

    # 将原始张量复制到新张量的对应位置
    padded_tensor[new_slices] = tensor[orig_slices]

    return padded_tensor


def calculate_weight(patches, weight, key, intermediate_dtype=torch.float32, original_weights=None):
    """
    根据补丁信息调整权重。

    该函数遍历提供的补丁列表，对权重应用不同的补丁操作（如差分更新、设置权重等）。
    每个补丁包含强度、值、模型强度、偏移量和函数等信息，用于精确控制权重的调整方式。

    参数:
        patches (list): 补丁列表，每个补丁是一个包含以下元素的元组：
            - strength (float): 补丁的强度因子，用于控制补丁应用的程度。
            - v (object): 补丁的值，可以是张量、列表或权重适配器对象。
            - strength_model (float): 模型强度因子，用于调整模型权重的强度。
            - offset (tuple, 可选): 偏移量，用于选择权重的子区域。
            - function (callable, 可选): 应用补丁前对补丁值进行转换的函数。
        weight (torch.Tensor): 要调整的原始权重张量。
        key (str): 当前权重的键名，用于日志记录和调试。
        intermediate_dtype (torch.dtype, 可选): 中间计算使用的数据类型，默认为 torch.float32。
        original_weights (dict, 可选): 原始权重字典，用于某些补丁操作，默认为 None。

    返回:
        torch.Tensor: 调整后的权重张量。
    """
    for p in patches:

        # 解包补丁信息
        strength = p[0]          # 补丁强度因子
        v = p[1]                 # 补丁值
        strength_model = p[2]    # 模型强度因子
        offset = p[3]            # 偏移量
        function = p[4]          # 应用补丁前对补丁值进行转换的函数

        # 如果未提供函数，则使用恒等函数
        if function is None:
            function = lambda a: a

        old_weight = None
        # 如果提供了偏移量，则对权重进行切片
        if offset is not None:
            old_weight = weight
            weight = weight.narrow(offset[0], offset[1], offset[2])

        # 如果模型强度因子不为1.0，则调整权重
        if strength_model != 1.0:
            weight *= strength_model

        # 如果补丁值是列表，则递归调用 calculate_weight 进行处理
        if isinstance(v, list):
            v = (calculate_weight(v[1:], v[0][1](model_management.cast_to_device(v[0][0], weight.device, intermediate_dtype, copy=True), inplace=True), key, intermediate_dtype=intermediate_dtype), )

        # 如果补丁值是权重适配器对象，则使用适配器的 calculate_weight 方法进行处理
        if isinstance(v, weight_adapter.WeightAdapterBase):
            output = v.calculate_weight(weight, key, strength, strength_model, offset, function, intermediate_dtype, original_weights)
            if output is None:
                logging.warning("Calculate Weight Failed: {} {}".format(v.name, key))
            else:
                weight = output
                if old_weight is not None:
                    weight = old_weight
            continue
        
        # 如果补丁值的长度为1，则默认补丁类型为 "diff"
        if len(v) == 1:
            patch_type = "diff"
        
        # 如果补丁值的长度为2，则第一个元素是补丁类型，第二个元素是补丁值
        elif len(v) == 2:
            patch_type = v[0]
            v = v[1]

        # 处理 "diff" 类型的补丁
        if patch_type == "diff":
            diff: torch.Tensor = v[0]
            # 如果提供了额外的标志，则根据标志决定是否填充权重
            do_pad_weight = len(v) > 1 and v[1]['pad_weight']
            if do_pad_weight and diff.shape != weight.shape:
                logging.info("Pad weight {} from {} to shape: {}".format(key, weight.shape, diff.shape))
                weight = pad_tensor_to_shape(weight, diff.shape)

            # 如果强度不为0，则应用差分补丁
            if strength != 0.0:
                if diff.shape != weight.shape:
                    logging.warning("WARNING SHAPE MISMATCH {} WEIGHT NOT MERGED {} != {}".format(key, diff.shape, weight.shape))
                else:
                    # 将差分补丁转换为与权重相同设备和数据类型，并应用强度因子
                    weight += function(strength * model_management.cast_to_device(diff, weight.device, weight.dtype))
        
        # 处理 "set" 类型的补丁
        elif patch_type == "set":
            # 直接将权重设置为补丁值
            weight.copy_(v[0])

        # 处理 "model_as_lora" 类型的补丁
        elif patch_type == "model_as_lora":
            target_weight: torch.Tensor = v[0]
            # 计算目标权重与原始权重的差分
            diff_weight = model_management.cast_to_device(target_weight, weight.device, intermediate_dtype) - \
                          model_management.cast_to_device(original_weights[key][0][0], weight.device, intermediate_dtype)
            # 应用差分补丁
            weight += function(strength * model_management.cast_to_device(diff_weight, weight.device, weight.dtype))
        else:
            # 如果补丁类型无法识别，则记录警告
            logging.warning("patch type not recognized {} {}".format(patch_type, key))

        # 如果之前对权重进行了切片，则恢复原始权重
        if old_weight is not None:
            weight = old_weight

    return weight
