import torch
import torch.distributed as dist

def setup_ddp():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

import torch
import torch.nn as nn
from collections import OrderedDict
import os

def model_summary_to_string(model, input_tensor):
    """
    生成模型的详细结构摘要，包括每层名称、类型、输入/输出形状和参数量。
    
    Args:
        model (nn.Module): 要分析的 PyTorch 模型
        input_tensor (torch.Tensor): 示例输入张量，用于推断各层输入/输出形状
        
    Returns:
        str: 格式化的模型摘要字符串
    """
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = f"{class_name}-{module_idx}"
            if hasattr(module, 'weight') and hasattr(module.weight, 'size'):
                num_params = torch.prod(torch.tensor(module.weight.size())).item()
                if hasattr(module, 'bias') and module.bias is not None:
                    num_params += torch.prod(torch.tensor(module.bias.size())).item()
            else:
                num_params = 0

            # === 修复：安全提取 tensor 的 shape ===
            def extract_shapes(tensors):
                if isinstance(tensors, (list, tuple)):
                    shapes = []
                    for t in tensors:
                        if torch.is_tensor(t):
                            shapes.append(list(t.shape))
                        # 忽略非 tensor（如 (T,H,W) 这样的元组）
                    return shapes if shapes else "N/A"
                elif torch.is_tensor(tensors):
                    return list(tensors.shape)
                else:
                    return "N/A"

            input_shape = extract_shapes(input)
            output_shape = extract_shapes(output)
            # ===================================

            summary[m_key] = {
                "layer_name": m_key,
                "module_type": class_name,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "num_params": int(num_params)
            }

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model):
            hooks.append(module.register_forward_hook(hook))

    # 确保模型在 eval 模式
    device = next(model.parameters()).device
    model.eval()
    input_tensor = input_tensor.to(device)

    # 存储摘要信息
    summary = OrderedDict()
    hooks = []

    # 注册 hook
    model.apply(register_hook)

    # 前向传播
    with torch.no_grad():
        _ = model(input_tensor)

    # 移除 hooks
    for h in hooks:
        h.remove()

    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 构建字符串
    summary_str = "=" * 80 + "\n"
    summary_str += f"{'Layer Name':<30} {'Type':<20} {'Input Shape':<25} {'Output Shape':<25} {'Params':<12}\n"
    summary_str += "=" * 80 + "\n"

    for layer in summary.values():
        name = layer["layer_name"]
        typ = layer["module_type"]
        in_shape = str(layer["input_shape"])
        out_shape = str(layer["output_shape"])
        params = f"{layer['num_params']:,}"

        summary_str += f"{name:<30} {typ:<20} {in_shape:<25} {out_shape:<25} {params:<12}\n"

    summary_str += "=" * 80 + "\n"
    summary_str += f"Total Parameters: {total_params:,}\n"
    summary_str += f"Trainable Parameters: {trainable_params:,}\n"
    summary_str += f"Model Size (FP32): {total_params * 4 / (1024**2):.2f} MB\n"
    summary_str += "=" * 80 + "\n"

    return summary_str


def save_model_summary_to_txt(summary_str, save_path):
    """
    将模型摘要字符串保存到指定路径的 .txt 文件中。
    
    Args:
        summary_str (str): 由 model_summary_to_string 生成的摘要字符串
        save_path (str): 保存路径，例如 "/path/to/model_summary.txt"
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(summary_str)
    print(f"✅ Model summary saved to: {save_path}")